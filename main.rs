use actix_web::{web, App, HttpServer, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

// Memory structure for user interactions
#[derive(Serialize, Deserialize, Clone)]
struct UserMemory {
    topics: Vec<String>,
    questions: HashMap<String, u32>,
    preferences: UserPreferences,
    named_examples: HashMap<String, BeamExample>,
}

#[derive(Serialize, Deserialize, Clone)]
struct UserPreferences {
    explanation_style: String,
    humor_level: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct BeamExample {
    name: String,
    beam_type: String,
    load: f64,
    length: f64,
    material: String,
    timestamp: u64,
}

// AI Response structure
#[derive(Serialize, Deserialize)]
struct AIResponse {
    question: String,
    answer: String,
    visualization: Option<VisualizationData>,
    emotional_context: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct VisualizationData {
    beam_type: String,
    html: String,
    formulas: HashMap<String, String>,
    calculations: Vec<String>,
}

// Global memory storage
struct AppState {
    memory: Mutex<HashMap<String, UserMemory>>,
}

impl Default for UserMemory {
    fn default() -> Self {
        Self {
            topics: Vec::new(),
            questions: HashMap::new(),
            preferences: UserPreferences {
                explanation_style: "detailed".to_string(),
                humor_level: "medium".to_string(),
            },
            named_examples: HashMap::new(),
        }
    }
}

// Emotional responses
const GREETINGS: &[&str] = &[
    "Hello! Ready to explore solid mechanics? 🌟",
    "Hey there! Great to see you learning engineering! 🚀",
    "Hi! Let's dive into some fascinating mechanics concepts! 💡",
];

const ENCOURAGEMENTS: &[&str] = &[
    "You're asking great questions! This is how engineers think!",
    "Excellent question! You're really understanding the material.",
    "I love your curiosity! This is exactly the right mindset for engineering.",
];

const JOKES: &[&str] = &[
    "Why did the beam go to therapy? It had too much bending moment!",
    "What's a beam's favorite type of music? Heavy metal, of course!",
    "Why was the math book sad? It had too many problems!",
];

// Knowledge base for beam bending
const BEAM_KNOWLEDGE: &[(&str, &str)] = &[
    (
        "bending equation",
        "The fundamental bending equation is: M/I = σ/y = E/R\nWhere:\n- M = Bending moment\n- I = Moment of inertia\n- σ = Bending stress\n- y = Distance from neutral axis\n- E = Modulus of elasticity\n- R = Radius of curvature"
    ),
    (
        "simply supported beam",
        "Simply Supported Beam:\n- Maximum bending moment: M_max = WL/4 at center\n- Maximum deflection: δ_max = WL³/(48EI) at center\n- Reactions: R_A = R_B = W/2"
    ),
    (
        "cantilever beam", 
        "Cantilever Beam:\n- Maximum bending moment: M_max = WL at fixed end\n- Maximum deflection: δ_max = WL³/(3EI) at free end\n- Reactions: R_A = W, M_A = WL"
    ),
    (
        "fixed beam",
        "Fixed Beam:\n- Maximum bending moment: M_max = WL/8 at ends and center\n- Maximum deflection: δ_max = WL³/(192EI) at center\n- Reactions: R_A = R_B = W/2"
    ),
];

async fn ask_question(
    data: web::Json<QuestionRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let question = data.question.clone();
    let user_id = data.user_id.clone().unwrap_or_else(|| "default".to_string());
    
    // Get or create user memory
    let mut memory_map = state.memory.lock().unwrap();
    let user_memory = memory_map.entry(user_id.clone()).or_insert_with(UserMemory::default);
    
    // Check for repetition
    let repetition_count = user_memory.questions.entry(question.clone()).or_insert(0);
    *repetition_count += 1;
    
    let mut response = String::new();
    
    // Add emotional context for repetition
    if *repetition_count >= 4 {
        response.push_str(&format!("I remember we've discussed \"{}\" before! Let me recall what we covered...\n\n", question));
    }
    
    // Add emotional response
    if let Some(emotional) = get_emotional_response(&question) {
        response.push_str(&format!("{}\n\n", emotional));
    }
    
    // Generate technical answer
    let technical_answer = generate_technical_answer(&question);
    response.push_str(&technical_answer);
    
    // Check if visualization is needed
    let visualization = if question.to_lowercase().contains("beam") || 
                           question.to_lowercase().contains("bending") ||
                           question.to_lowercase().contains("deflection") {
        Some(generate_beam_visualization("simply_supported", 1000.0, 2.0))
    } else {
        None
    };
    
    let ai_response = AIResponse {
        question,
        answer: response,
        visualization,
        emotional_context: None,
    };
    
    Ok(HttpResponse::Ok().json(ai_response))
}

fn get_emotional_response(question: &str) -> Option<String> {
    let question_lower = question.to_lowercase();
    
    // Check for greetings
    if question_lower.contains("hello") || question_lower.contains("hi") || 
       question_lower.contains("hey") || question_lower.contains("greetings") {
        return Some(GREETINGS[rand::random::<usize>() % GREETINGS.len()].to_string());
    }
    
    // Check for frustration
    if question_lower.contains("hard") || question_lower.contains("difficult") ||
       question_lower.contains("confused") || question_lower.contains("help") {
        let encouragement = ENCOURAGEMENTS[rand::random::<usize>() % ENCOURAGEMENTS.len()];
        let joke = JOKES[rand::random::<usize>() % JOKES.len()];
        return Some(format!("{} {}", encouragement, joke));
    }
    
    // Occasionally add a joke
    if rand::random::<f32>() < 0.2 {
        return Some(JOKES[rand::random::<usize>() % JOKES.len()].to_string());
    }
    
    None
}

fn generate_technical_answer(question: &str) -> String {
    let question_lower = question.to_lowercase();
    
    // Search knowledge base
    for (keyword, answer) in BEAM_KNOWLEDGE {
        if question_lower.contains(keyword) {
            return answer.to_string();
        }
    }
    
    // Fallback responses
    if question_lower.contains("formula") {
        return "I can help with various solid mechanics formulas! Try asking about:\n- Bending moment formulas\n- Deflection equations\n- Stress calculations\n- Beam reactions".to_string();
    }
    
    if question_lower.contains("calculate") {
        return "I can perform calculations for:\n- Beam bending moments\n- Deflections\n- Stress distributions\n- Support reactions\nPlease provide the specific values you'd like to calculate!".to_string();
    }
    
    "I'm your Solid Mechanics AI tutor! I specialize in beam bending, stress analysis, and mechanical design. Ask me about:\n- Bending of beams\n- Stress and strain\n- Material properties\n- Engineering calculations\n- Real-world applications".to_string()
}

fn generate_beam_visualization(beam_type: &str, load: f64, length: f64) -> VisualizationData {
    let (html, formulas, calculations) = match beam_type {
        "simply_supported" => (
            r#"<div class="beam-simulation">
                <div class="beam" style="width: 80%; height: 10px; left: 10%; top: 45%; background: #555;"></div>
                <div class="support" style="width: 20px; height: 30px; left: 10%;"></div>
                <div class="support" style="width: 20px; height: 30px; left: 90%;"></div>
                <div class="load" style="width: 10px; height: 50px; left: 50%; top: 0px; background: #ff4444;"></div>
            </div>"#.to_string(),
            HashMap::from([
                ("max_moment".to_string(), format!("M_max = W × L / 4 = {:.1} N·m", load * length / 4.0)),
                ("max_deflection".to_string(), "δ_max = W × L³ / (48 × E × I)".to_string()),
                ("reactions".to_string(), format!("R_A = R_B = W / 2 = {:.1} N", load / 2.0)),
            ]),
            vec![
                format!("1. Calculate reactions: R = W/2 = {:.1}N/2 = {:.1}N", load, load/2.0),
                format!("2. Maximum bending moment: M = WL/4 = {:.1}N × {:.1}m / 4 = {:.1} N·m", load, length, load*length/4.0),
                "3. Maximum deflection: δ = WL³/(48EI)".to_string(),
            ],
        ),
        "cantilever" => (
            r#"<div class="beam-simulation">
                <div class="beam" style="width: 70%; height: 10px; left: 10%; top: 45%; background: #555;"></div>
                <div class="support" style="width: 30px; height: 40px; left: 10%;"></div>
                <div class="load" style="width: 10px; height: 50px; left: 80%; top: 0px; background: #ff4444;"></div>
            </div>"#.to_string(),
            HashMap::from([
                ("max_moment".to_string(), format!("M_max = W × L = {:.1} N·m", load * length)),
                ("max_deflection".to_string(), "δ_max = W × L³ / (3 × E × I)".to_string()),
                ("reactions".to_string(), format!("R_A = W = {:.1} N, M_A = W × L = {:.1} N·m", load, load * length)),
            ]),
            vec![
                format!("1. Fixed end reaction: R = W = {:.1}N", load),
                format!("2. Fixed end moment: M = WL = {:.1}N × {:.1}m = {:.1} N·m", load, length, load*length),
                "3. Maximum deflection at free end: δ = WL³/(3EI)".to_string(),
            ],
        ),
        _ => (
            r#"<div class="beam-simulation">
                <div class="beam" style="width: 80%; height: 10px; left: 10%; top: 45%; background: #555;"></div>
                <div class="support" style="width: 25px; height: 35px; left: 10%;"></div>
                <div class="support" style="width: 25px; height: 35px; left: 90%;"></div>
                <div class="load" style="width: 10px; height: 50px; left: 50%; top: 0px; background: #ff4444;"></div>
            </div>"#.to_string(),
            HashMap::from([
                ("max_moment".to_string(), format!("M_max = W × L / 8 = {:.1} N·m", load * length / 8.0)),
                ("max_deflection".to_string(), "δ_max = W × L³ / (192 × E × I)".to_string()),
                ("reactions".to_string(), format!("R_A = R_B = W / 2 = {:.1} N", load / 2.0)),
            ]),
            vec![
                format!("1. Reactions at both ends: R = W/2 = {:.1}N/2 = {:.1}N", load, load/2.0),
                format!("2. Fixed end moments: M = WL/8 = {:.1}N × {:.1}m / 8 = {:.1} N·m", load, length, load*length/8.0),
                "3. Maximum deflection at center: δ = WL³/(192EI)".to_string(),
            ],
        ),
    };
    
    VisualizationData {
        beam_type: beam_type.to_string(),
        html,
        formulas,
        calculations,
    }
}

#[derive(Deserialize)]
struct QuestionRequest {
    question: String,
    user_id: Option<String>,
}

#[derive(Deserialize)]
struct BeamVisualizationRequest {
    beam_type: String,
    load: Option<f64>,
    length: Option<f64>,
    user_id: Option<String>,
}

async fn get_beam_visualization(
    data: web::Json<BeamVisualizationRequest>,
) -> Result<HttpResponse> {
    let beam_type = data.beam_type.clone();
    let load = data.load.unwrap_or(1000.0);
    let length = data.length.unwrap_or(2.0);
    
    let visualization = generate_beam_visualization(&beam_type, load, length);
    
    Ok(HttpResponse::Ok().json(visualization))
}

async fn get_user_memory(
    user_id: web::Path<String>,
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let memory_map = state.memory.lock().unwrap();
    
    if let Some(user_memory) = memory_map.get(&user_id.into_inner()) {
        Ok(HttpResponse::Ok().json(user_memory))
    } else {
        Ok(HttpResponse::NotFound().body("User not found"))
    }
}

async fn save_example(
    data: web::Json<BeamExample>,
    state: web::Data<AppState>,
) -> Result<HttpResponse> {
    let example = data.into_inner();
    let user_id = "default".to_string(); // In real implementation, get from auth
    
    let mut memory_map = state.memory.lock().unwrap();
    let user_memory = memory_map.entry(user_id).or_insert_with(UserMemory::default);
    
    // Add timestamp if not present
    let example_with_timestamp = BeamExample {
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        ..example
    };
    
    user_memory.named_examples.insert(example_with_timestamp.name.clone(), example_with_timestamp);
    
    Ok(HttpResponse::Ok().json("Example saved successfully"))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Solid Mechanics AI Tutor starting...");
    println!("Available endpoints:");
    println!("  POST /api/ask - Ask questions about solid mechanics");
    println!("  POST /api/visualization/beam - Get beam visualization");
    println!("  GET /api/memory/{user_id} - Get user memory");
    println!("  POST /api/save-example - Save beam example");
    
    let app_state = web::Data::new(AppState {
        memory: Mutex::new(HashMap::new()),
    });
    
    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/api/ask", web::post().to(ask_question))
            .route("/api/visualization/beam", web::post().to(get_beam_visualization))
            .route("/api/memory/{user_id}", web::get().to(get_user_memory))
            .route("/api/save-example", web::post().to(save_example))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emotional_responses() {
        assert!(get_emotional_response("hello").is_some());
        assert!(get_emotional_response("this is hard").is_some());
    }

    #[test]
    fn test_technical_answers() {
        let answer = generate_technical_answer("bending equation");
        assert!(answer.contains("bending equation"));
        
        let answer = generate_technical_answer("simply supported beam");
        assert!(answer.contains("Simply Supported Beam"));
    }
}