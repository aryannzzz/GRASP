"""
LLM Interface Module
Handles communication with GPT-4 for action prediction.
"""

import os
from typing import List, Dict, Tuple
import yaml
from pathlib import Path


class LLMInterface:
    """
    Interface for querying LLMs (GPT-4) for robot action prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize LLM interface.
        
        Args:
            config: Configuration dictionary
        """
        icl_config = config['icl']
        
        self.model = icl_config['llm_model']
        self.temperature = icl_config['temperature']
        self.max_tokens = icl_config['max_tokens']
        
        # Get API key from environment
        api_key_env = icl_config['api_key_env']
        self.api_key = os.environ.get(api_key_env)
        
        if not self.api_key:
            print(f"WARNING: {api_key_env} not found in environment variables!")
            print(f"Please set it with: export {api_key_env}=your_api_key")
        
        # Initialize client
        self._init_client()
        
        print(f"Initialized LLMInterface:")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                print("  OpenAI client initialized")
            else:
                self.client = None
                print("  WARNING: OpenAI client not initialized (no API key)")
                
        except ImportError:
            print("Error: openai library not installed!")
            print("Install with: pip install openai")
            self.client = None
    
    def query(self, system_message: str, user_prompt: str) -> str:
        """
        Query the LLM with a prompt.
        
        Args:
            system_message: System message defining the task
            user_prompt: User prompt with demonstrations and test input
            
        Returns:
            response: LLM response text
        """
        if self.client is None:
            print("Error: LLM client not initialized!")
            return ""
        
        try:
            # Query OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response = completion.choices[0].message.content
            
            # Log usage
            if hasattr(completion, 'usage'):
                print(f"  Tokens used: {completion.usage.total_tokens}")
            
            return response
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return ""
    
    def predict_actions(self,
                       system_message: str,
                       user_prompt: str,
                       parse_fn=None) -> Tuple[List, List]:
        """
        Predict actions using LLM and parse the response.
        
        Args:
            system_message: System message
            user_prompt: User prompt
            parse_fn: Function to parse LLM response (optional)
            
        Returns:
            actions: List of predicted discrete actions
            grippers: List of predicted gripper states
        """
        # Query LLM
        response = self.query(system_message, user_prompt)
        
        if not response:
            return [], []
        
        # Parse response if parser provided
        if parse_fn is not None:
            actions, grippers = parse_fn(response)
            return actions, grippers
        
        return response, None
    
    def query_batch(self,
                   system_messages: List[str],
                   user_prompts: List[str]) -> List[str]:
        """
        Query LLM with batch of prompts.
        
        Args:
            system_messages: List of system messages
            user_prompts: List of user prompts
            
        Returns:
            responses: List of LLM responses
        """
        responses = []
        
        for system_msg, user_prompt in zip(system_messages, user_prompts):
            response = self.query(system_msg, user_prompt)
            responses.append(response)
        
        return responses


def test_llm_interface():
    """Test LLM interface with a simple query."""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create interface
    llm = LLMInterface(config)
    
    # Test query
    system_msg = "You are a helpful assistant that predicts robot actions."
    
    user_prompt = """Given these demonstrations:
    
Demo 1:
Input: {[tape]: [10, 20, 30, 5, 10, 15], [Task]: pick tape}
Output: {[15, 25, 35, 8, 12, 18, 0], [10, 20, 30, 5, 10, 15, 1]}

Demo 2:
Input: {[tape]: [12, 22, 32, 6, 11, 16], [Task]: pick tape}
Output: {[17, 27, 37, 9, 13, 19, 0], [12, 22, 32, 6, 11, 16, 1]}

Test Input: {[tape]: [11, 21, 31, 5, 10, 16], [Task]: pick tape}

Predict the output for the test input."""
    
    print("Querying LLM...")
    response = llm.query(system_msg, user_prompt)
    
    print("\nLLM Response:")
    print(response)


if __name__ == "__main__":
    test_llm_interface()
