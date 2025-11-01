# solid_mechanics_ai.py
import torch
import torch.nn as nn
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF for PDF processing
import json
import random

class SolidMechanicsAI:
    def __init__(self):
        self.qa_pipeline = None
        self.vectorizer = None
        self.corpus = []
        self.qa_pairs = []
        self.setup_models()
        self.load_pdf_content()
        
    def setup_models(self):
        """Initialize the AI models"""
        try:
            # Use a smaller, more efficient model for this domain
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
        except:
            # Fallback to rule-based system if model fails to load
            self.qa_pipeline = None
            
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        
    def load_pdf_content(self):
        """Extract and process content from the solid mechanics PDF"""
        pdf_content = """
        TORSION:
        - Torsion formula: T = (π/16) × τ × d³
        - For solid shafts: T = (π/16) × τ × d³
        - For hollow shafts: T = (π/16) × τ × (d₀⁴ - dᵢ⁴)/d₀
        - Shear stress: τ = (16T)/(πd³)
        - Polar modulus = Polar moment of inertia / Radius
        
        SPRINGS:
        - Spring deflection: δ = (64WR³n)/(Gd⁴)
        - Spring stiffness: k = W/δ = (Gd⁴)/(64R³n)
        - Energy stored: U = (1/2)Wδ
        - Shear stress in spring: τ = (16WR)/(πd³)
        
        PRESSURE VESSELS:
        - Circumferential stress: σ_c = (Pd)/(2t)
        - Longitudinal stress: σ_L = (Pd)/(4t)
        - Thin-walled assumption: t << d
        
        PRINCIPAL STRESSES:
        - Principal stresses: σ₁,₂ = (σ_x + σ_y)/2 ± √[((σ_x - σ_y)/2)² + τ_xy²]
        - Maximum shear stress: τ_max = √[((σ_x - σ_y)/2)² + τ_xy²]
        - Angle: tan(2θ) = (2τ_xy)/(σ_x - σ_y)
        
        BENDING OF BEAMS:
        - Bending equation: M/I = σ/y = E/R
        - Bending stress: σ = My/I
        - Section modulus: Z = I/y_max
        - Maximum bending stress: σ_max = M/Z
        - Deflection equations for different beam types
        - Shear force and bending moment diagrams
        
        STRAIN ENERGY:
        - Strain energy in tension: U = (σ²V)/(2E)
        - Volumetric strain: ΔV/V = (σ_x + σ_y + σ_z)(1-2ν)/E
        
        MATERIAL PROPERTIES:
        - Young's modulus (E), Shear modulus (G), Poisson's ratio (ν)
        - Relationship: G = E/(2(1+ν))
        
        ASSUMPTIONS IN BENDING:
        1. Beam is initially straight
        2. Material is homogeneous and isotropic
        3. Hooke's law is valid
        4. Plane sections remain plane after bending
        
        EXAMPLES:
        - For τ=46 N/mm², d=250mm: T=(π/16)×46×(250)³=141.05 N/mm
        - Spring with W=200N, R=60mm, n=10, d=10mm, G=80×10³ N/mm²:
          δ=(64×200×60³×10)/(80×10³×10⁴)=34.56mm
          k=200/34.56=5.78 N/mm
          U=0.5×200×34.56=3456 N·mm
        - Simply supported beam with central load W=1000N, L=2m:
          M_max = WL/4 = 1000×2/4 = 500 N·m
          δ_max = WL³/(48EI)
        """
        
        # Process the content into QA pairs
        self.process_content_to_qa(pdf_content)
        
    def process_content_to_qa(self, content):
        """Convert PDF content into question-answer pairs"""
        lines = content.split('\n')
        current_topic = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.endswith(':'):
                current_topic = line[:-1]
            elif ':' in line and '=' in line:
                # Formula lines
                parts = line.split(':', 1)
                if len(parts) == 2:
                    concept = parts[0].strip()
                    formula = parts[1].strip()
                    self.qa_pairs.append({
                        'question': f'What is the formula for {concept}?',
                        'answer': f'The formula for {concept} is: {formula}'
                    })
            elif '-' in line and len(line) > 10:
                # Concept lines
                concept = line[1:].strip()
                self.qa_pairs.append({
                    'question': f'Explain {concept}',
                    'answer': f'{concept} is an important concept in solid mechanics. {self.generate_explanation(concept)}'
                })
        
        # Add the corpus for similarity search
        self.corpus = [pair['answer'] for pair in self.qa_pairs]
        if self.corpus:
            self.vectorizer.fit(self.corpus)
    
    def generate_explanation(self, concept):
        """Generate explanations for concepts"""
        explanations = {
            'torsion': 'Torsion refers to the twisting of an object due to an applied torque. It is crucial in shaft design.',
            'springs': 'Springs are mechanical components that store and release energy, commonly used for shock absorption.',
            'pressure vessels': 'Pressure vessels are containers designed to hold gases or liquids at high pressures.',
            'principal stresses': 'Principal stresses are the maximum and minimum normal stresses at a point.',
            'bending of beams': 'Bending of beams deals with the behavior of structural elements subjected to transverse loads.',
            'strain energy': 'Strain energy is the energy stored in a material when it is deformed elastically.'
        }
        
        for key, explanation in explanations.items():
            if key in concept.lower():
                return explanation
        return "This is a fundamental concept in solid mechanics engineering."
    
    def find_similar_question(self, question):
        """Find the most similar question using TF-IDF"""
        if not self.corpus:
            return None
            
        question_vec = self.vectorizer.transform([question])
        corpus_vec = self.vectorizer.transform(self.corpus)
        
        similarities = cosine_similarity(question_vec, corpus_vec)[0]
        best_match_idx = np.argmax(similarities)
        
        if similarities[best_match_idx] > 0.3:  # Threshold
            return self.qa_pairs[best_match_idx]
        return None
    
    def get_emotional_response(self, question, user_memory):
        """Generate emotional and motivational responses"""
        greetings = [
            "Hello! Ready to explore solid mechanics?",
            "Hey there! Great to see you learning engineering!",
            "Hi! Let's dive into some fascinating mechanics concepts!"
        ]
        
        encouragements = [
            "You're asking great questions! This is how engineers think!",
            "Excellent question! You're really understanding the material.",
            "I love your curiosity! This is exactly the right mindset for engineering."
        ]
        
        jokes = [
            "Why did the beam go to therapy? It had too much bending moment!",
            "What's a beam's favorite type of music? Heavy metal, of course!",
            "Why was the math book sad? It had too many problems!"
        ]
        
        # Check if this is a greeting
        if any(word in question.lower() for word in ['hello', 'hi', 'hey', 'greetings']):
            return random.choice(greetings)
        
        # Check if user seems frustrated
        if any(word in question.lower() for word in ['hard', 'difficult', 'confused', 'help']):
            return random.choice(encouragements) + " " + random.choice(jokes)
        
        # Occasionally add a joke for motivation
        if random.random() < 0.2:  # 20% chance
            return random.choice(jokes)
        
        return None
    
    def ask_question(self, question, user_memory=None, repetition_message=None):
        """Answer questions about solid mechanics with emotional intelligence"""
        # First, check for emotional response
        emotional_response = self.get_emotional_response(question, user_memory)
        
        # Handle repetition
        if repetition_message:
            response = repetition_message + "\n\n"
        else:
            response = ""
            
        if emotional_response:
            response += emotional_response + "\n\n"
        
        # First try to find similar question in our knowledge base
        similar_qa = self.find_similar_question(question)
        if similar_qa:
            return response + similar_qa['answer']
        
        # If no good match, use the QA pipeline
        if self.qa_pipeline:
            try:
                # Create context from our knowledge
                context = " ".join([qa['answer'] for qa in self.qa_pairs[:10]])
                result = self.qa_pipeline(question=question, context=context)
                return response + result['answer']
            except:
                pass
        
        # Fallback response
        return response + "I'm trained in solid mechanics. I can help with torsion, springs, pressure vessels, stress analysis, beam bending, and material properties. Could you please rephrase your question?"
    
    def calculate_formula(self, formula_name, **kwargs):
        """Calculate engineering formulas"""
        calculations = {
            'torsion_solid': lambda: (np.pi/16) * kwargs.get('tau', 0) * (kwargs.get('d', 0)**3),
            'torsion_hollow': lambda: (np.pi/16) * kwargs.get('tau', 0) * ((kwargs.get('d0', 0)**4 - kwargs.get('di', 0)**4) / kwargs.get('d0', 1)),
            'spring_deflection': lambda: (64 * kwargs.get('W', 0) * (kwargs.get('R', 0)**3) * kwargs.get('n', 0)) / (kwargs.get('G', 1) * (kwargs.get('d', 1)**4)),
            'circumferential_stress': lambda: (kwargs.get('P', 0) * kwargs.get('d', 0)) / (2 * kwargs.get('t', 1)),
            'longitudinal_stress': lambda: (kwargs.get('P', 0) * kwargs.get('d', 0)) / (4 * kwargs.get('t', 1)),
            'principal_stress_max': lambda: (kwargs.get('sigma_x', 0) + kwargs.get('sigma_y', 0))/2 + np.sqrt(((kwargs.get('sigma_x', 0) - kwargs.get('sigma_y', 0))/2)**2 + kwargs.get('tau_xy', 0)**2),
            'principal_stress_min': lambda: (kwargs.get('sigma_x', 0) + kwargs.get('sigma_y', 0))/2 - np.sqrt(((kwargs.get('sigma_x', 0) - kwargs.get('sigma_y', 0))/2)**2 + kwargs.get('tau_xy', 0)**2),
            'bending_moment_ssb_central': lambda: (kwargs.get('W', 0) * kwargs.get('L', 0)) / 4,
            'deflection_ssb_central': lambda: (kwargs.get('W', 0) * (kwargs.get('L', 0)**3)) / (48 * kwargs.get('E', 1) * kwargs.get('I', 1)),
            'bending_moment_cantilever': lambda: kwargs.get('W', 0) * kwargs.get('L', 0),
            'deflection_cantilever': lambda: (kwargs.get('W', 0) * (kwargs.get('L', 0)**3)) / (3 * kwargs.get('E', 1) * kwargs.get('I', 1))
        }
        
        if formula_name in calculations:
            try:
                result = calculations[formula_name]()
                return f"The calculated result is: {result:.4f}"
            except:
                return "Error in calculation. Please check the input parameters."
        else:
            return "Formula not recognized."

# Flask API for web integration
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ai_model = SolidMechanicsAI()

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question', '')
    user_memory = data.get('memory', {})
    repetition_message = data.get('repetition', None)
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    answer = ai_model.ask_question(question, user_memory, repetition_message)
    return jsonify({'question': question, 'answer': answer})

@app.route('/api/calculate', methods=['POST'])
def calculate():
    data = request.json
    formula = data.get('formula', '')
    parameters = data.get('parameters', {})
    
    if not formula:
        return jsonify({'error': 'No formula specified'}), 400
    
    result = ai_model.calculate_formula(formula, **parameters)
    return jsonify({'formula': formula, 'result': result})

@app.route('/api/topics', methods=['GET'])
def get_topics():
    topics = [
        "Torsion and Shaft Design",
        "Springs and Deflection", 
        "Pressure Vessels",
        "Principal Stresses",
        "Bending of Beams",
        "Strain Energy",
        "Material Properties"
    ]
    return jsonify({'topics': topics})

@app.route('/api/visualization/beam', methods=['POST'])
def get_beam_visualization():
    data = request.json
    beam_type = data.get('type', 'simply_supported')
    load = data.get('load', 1000)
    length = data.get('length', 2)
    
    visualization_data = {
        'type': beam_type,
        'load': load,
        'length': length,
        'html': generate_beam_html(beam_type, load, length),
        'formulas': get_beam_formulas(beam_type, load, length)
    }
    
    return jsonify(visualization_data)

def generate_beam_html(beam_type, load, length):
    """Generate HTML for beam visualization"""
    if beam_type == 'simply_supported':
        return f"""
        <div class="beam-simulation">
            <div class="beam" style="width: 80%; height: 10px; left: 10%; top: 45%; background: #555;"></div>
            <div class="support" style="width: 20px; height: 30px; left: 10%;"></div>
            <div class="support" style="width: 20px; height: 30px; left: 90%;"></div>
            <div class="load" style="width: 10px; height: 50px; left: 50%; top: 0px; background: #ff4444;"></div>
        </div>
        """
    elif beam_type == 'cantilever':
        return f"""
        <div class="beam-simulation">
            <div class="beam" style="width: 70%; height: 10px; left: 10%; top: 45%; background: #555;"></div>
            <div class="support" style="width: 30px; height: 40px; left: 10%;"></div>
            <div class="load" style="width: 10px; height: 50px; left: 80%; top: 0px; background: #ff4444;"></div>
        </div>
        """
    else:  # fixed
        return f"""
        <div class="beam-simulation">
            <div class="beam" style="width: 80%; height: 10px; left: 10%; top: 45%; background: #555;"></div>
            <div class="support" style="width: 25px; height: 35px; left: 10%;"></div>
            <div class="support" style="width: 25px; height: 35px; left: 90%;"></div>
            <div class="load" style="width: 10px; height: 50px; left: 50%; top: 0px; background: #ff4444;"></div>
        </div>
        """

def get_beam_formulas(beam_type, load, length):
    """Get formulas for different beam types"""
    if beam_type == 'simply_supported':
        return {
            'max_moment': f'M_max = W × L / 4 = {load} × {length} / 4 = {load * length / 4} N·m',
            'max_deflection': f'δ_max = W × L³ / (48 × E × I)',
            'reactions': f'R_A = R_B = W / 2 = {load} / 2 = {load / 2} N'
        }
    elif beam_type == 'cantilever':
        return {
            'max_moment': f'M_max = W × L = {load} × {length} = {load * length} N·m',
            'max_deflection': f'δ_max = W × L³ / (3 × E × I)',
            'reactions': f'R_A = W = {load} N, M_A = W × L = {load * length} N·m'
        }
    else:  # fixed
        return {
            'max_moment': f'M_max = W × L / 8 = {load} × {length} / 8 = {load * length / 8} N·m',
            'max_deflection': f'δ_max = W × L³ / (192 × E × I)',
            'reactions': f'R_A = R_B = W / 2 = {load} / 2 = {load / 2} N'
        }

if __name__ == '__main__':
    print("Solid Mechanics AI Assistant Starting...")
    print("Available endpoints:")
    print("  POST /api/ask - Ask questions about solid mechanics")
    print("  POST /api/calculate - Perform engineering calculations") 
    print("  GET /api/topics - Get available topics")
    print("  POST /api/visualization/beam - Get beam visualization data")
    
    app.run(debug=True, port=5000)