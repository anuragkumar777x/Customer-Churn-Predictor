import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def verify_model_availability():
    """Verify if the model is available and supported."""
    return False, "Chat service temporarily unavailable"

def test_api_connection():
    """Test the Google API connection and key validity."""
    return False, "Chat service temporarily unavailable"

class CustomerRetentionChatbot:
    def __init__(self):
        """Initialize the chatbot with error handling."""
        self.is_available = False
        self.error_message = "We apologize, but our chat service is temporarily unavailable. Please try again later."
        
    def get_response(self, user_input, prediction_data=None):
        """Return a server error message when chatbot is unavailable."""
        return self.error_message
            
    def get_retention_strategies(self, prediction_data):
        """Return a server error message when chatbot is unavailable."""
        return self.error_message

    def verify_model_availability(self):
        """Verify if the model is available and supported."""
        try:
            models = genai.list_models()
            available_models = [model.name for model in models]
            if 'models/gemini-2.0-pro-exp' in available_models:
                return True, "Model is available"
            return False, f"Available models: {', '.join(available_models)}"
        except Exception as e:
            return False, f"Failed to list models: {str(e)}"

    def test_api_connection(self):
        """Test the Google API connection and key validity."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return False, "API key not found in environment variables"
            if len(api_key.strip()) < 10:  # Basic validation
                return False, "API key appears to be invalid"
            
            # Configure the API
            genai.configure(api_key=api_key)
            
            # Verify model availability
            model_available, model_message = self.verify_model_availability()
            if not model_available:
                return False, f"Model verification failed: {model_message}"
            
            # Create a test model
            model = genai.GenerativeModel('models/gemini-2.0-pro-exp')
            
            # Try a simple test message
            response = model.generate_content("Test connection")
            
            if response and response.text:
                return True, "API connection successful"
            return False, "API response was empty"
        except Exception as e:
            error_message = str(e)
            if "invalid api key" in error_message.lower():
                return False, "Invalid API key. Please check your API key in the .env file."
            return False, f"API connection failed: {error_message}"

    def configure_chatbot(self):
        """Configure the chatbot with API key validation and error handling."""
        # Validate and configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not found in environment variables. Please check your .env file.")
        if len(api_key.strip()) < 10:  # Basic validation
            raise ValueError("API key appears to be invalid. Please check your .env file.")
            
        try:
            genai.configure(api_key=api_key)
            
            # Verify model availability
            model_available, model_message = self.verify_model_availability()
            if not model_available:
                raise ValueError(f"Model verification failed: {model_message}")
                
        except Exception as e:
            raise ValueError(f"Failed to configure API: {str(e)}")
        
        # Set up the model with appropriate configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        
        try:
            self.model = genai.GenerativeModel(
                model_name='models/gemini-2.0-pro-exp',
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            raise ValueError(f"Failed to initialize chat model: {str(e)}")
        
        # System message to set the context
        self.system_message = """You are a helpful AI assistant specializing in customer retention and churn prediction. 
        Your role is to:
        1. Help users understand customer retention metrics and predictions
        2. Provide insights about why customers might churn
        3. Suggest strategies to improve customer retention
        4. Explain the prediction model's results in simple terms
        
        Always be professional, concise, and focused on actionable insights.
        
        When providing suggestions:
        - Be specific and practical
        - Consider the customer's profile
        - Focus on proven retention strategies
        - Explain the reasoning behind each suggestion
        """
        
        # Initialize the chat with the system message
        try:
            response = self.chat.send_message(self.system_message)
            if not response or not response.text:
                raise ValueError("Empty response from API during initialization")
        except Exception as e:
            raise ValueError(f"Failed to initialize chat with system message: {str(e)}")

    def get_response(self, user_input, prediction_data=None):
        """Get a response from the chatbot with enhanced context and error handling."""
        try:
            # Add prediction context if available
            if prediction_data:
                risk_level = "high" if prediction_data.get('probability', 0) > 0.5 else "low"
                context = f"""
                Current customer analysis:
                - Risk Level: {risk_level.title()} (Churn Probability: {prediction_data.get('probability', 0):.1%})
                - Profile: {prediction_data.get('gender', 'Unknown')} customer from {prediction_data.get('geography', 'Unknown')}
                - Age: {prediction_data.get('age', 'N/A')} years old
                - Credit Score: {prediction_data.get('credit_score', 'N/A')}
                - Products: {prediction_data.get('num_of_products', 'N/A')}
                - Status: {'Active' if prediction_data.get('is_active_member', False) else 'Inactive'} member
                
                Key risk factors identified:
                {', '.join(prediction_data.get('key_factors', ['No specific factors available']))}
                
                User question: {user_input}
                
                Provide a helpful, specific response focusing on actionable insights and clear explanations.
                """
                user_input = context

            # Get response from the model
            response = self.chat.send_message(user_input)
            
            # Validate response
            if not response or not response.text:
                return "I apologize, but I received an empty response. Please try asking your question again."
            
            return response.text.strip()
            
        except Exception as e:
            error_message = str(e)
            if "invalid api key" in error_message.lower():
                return "I apologize, but there seems to be an issue with the API key. Please contact support."
            elif "rate limit" in error_message.lower():
                return "I apologize, but we've hit the rate limit. Please try again in a moment."
            else:
                return f"I apologize, but I'm experiencing technical difficulties. Please try again. Error: {error_message}"

    def get_retention_strategies(self, prediction_data):
        if not prediction_data:
            return "No prediction data available to generate retention strategies."

        prompt = f"""
        Based on the following customer data:
        - Exit Probability: {prediction_data.get('probability', 'N/A'):.2%}
        - Key Factors: {', '.join(prediction_data.get('key_factors', ['No specific factors available']))}
        - Geography: {prediction_data.get('geography', 'N/A')}
        - Age: {prediction_data.get('age', 'N/A')}
        - Credit Score: {prediction_data.get('credit_score', 'N/A')}
        - Active Member: {'Yes' if prediction_data.get('is_active_member', False) else 'No'}
        - Number of Products: {prediction_data.get('num_of_products', 'N/A')}
        
        Please provide 3 specific, actionable strategies to improve customer retention for this case.
        """
        
        try:
            response = self.chat.send_message(prompt)
            if not response or not response.text:
                return "I apologize, but I received an empty response. Please try again."
            return response.text
        except Exception as e:
            return f"I apologize, but I'm currently experiencing technical difficulties. Please try again later. Error: {str(e)}" 