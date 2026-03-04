"""
In-Context Learning (ICL) Constructor Module
Builds ICL prompts for GPT-4 following RoboPrompt paper Section IV-C.
"""

import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import yaml


class ICLConstructor:
    """
    Constructs ICL prompts for robot action prediction.
    
    Following RoboPrompt paper format:
    - Input: [O1], [O2], ..., [Om], [I]
    - Output: [A_t2], [A_t3], ..., [A_tT]
    - Prompt: x1 > y1, x2 > y2, ..., xn > yn, xTest >
    """
    
    def __init__(self, config: Dict, discretizer):
        """
        Initialize ICL constructor.
        
        Args:
            config: Configuration dictionary
            discretizer: ActionDiscretizer instance for formatting
        """
        icl_config = config['icl']
        
        self.num_demonstrations = icl_config['num_demonstrations']
        self.llm_model = icl_config['llm_model']
        self.temperature = icl_config['temperature']
        self.max_tokens = icl_config['max_tokens']
        
        self.discretizer = discretizer
        
        print(f"Initialized ICLConstructor:")
        print(f"  Model: {self.llm_model}")
        print(f"  Num demonstrations: {self.num_demonstrations}")
        print(f"  Temperature: {self.temperature}")
    
    def construct_input(self,
                       object_poses: Dict[str, np.ndarray],
                       instruction: str) -> str:
        """
        Construct input portion of ICL example: {[O1], [O2], ..., [Om], [I]}
        
        Args:
            object_poses: Dictionary mapping object name to discrete pose
            instruction: Language instruction
            
        Returns:
            input_text: Formatted input string
        """
        # Format object observations
        observations = []
        for obj_name, pose in object_poses.items():
            # Ensure pose is discretized
            if isinstance(pose, np.ndarray) and pose.dtype != int:
                pose = self.discretizer.discretize_pose(pose)
            
            obs_text = f"[{obj_name}]: {self.discretizer.format_discrete_pose_as_text(pose)}"
            observations.append(obs_text)
        
        # Add instruction
        observations.append(f"[Task]: {instruction}")
        
        # Combine with proper formatting
        input_text = "{" + ", ".join(observations) + "}"
        
        return input_text
    
    def construct_output(self, 
                        discrete_actions: List[np.ndarray],
                        discrete_grippers: List[int]) -> str:
        """
        Construct output portion of ICL example: {[A_t2], [A_t3], ..., [A_tT]}
        
        Args:
            discrete_actions: List of discrete action poses at keyframes
            discrete_grippers: List of discrete gripper states at keyframes
            
        Returns:
            output_text: Formatted output string
        """
        # Format actions
        actions = []
        for action, gripper in zip(discrete_actions, discrete_grippers):
            action_text = self.discretizer.format_discrete_pose_as_text(action, gripper)
            actions.append(action_text)
        
        # Combine with proper formatting
        output_text = "{" + ", ".join(actions) + "}"
        
        return output_text
    
    def construct_demonstration(self,
                               demo_data: Dict) -> Tuple[str, str]:
        """
        Construct a single ICL demonstration (input-output pair).
        
        Args:
            demo_data: Dictionary with:
                - 'object_poses': Dict[str, np.ndarray]
                - 'instruction': str
                - 'actions': List[np.ndarray]
                - 'grippers': List[int]
                
        Returns:
            input_text: Formatted input
            output_text: Formatted output
        """
        input_text = self.construct_input(
            demo_data['object_poses'],
            demo_data['instruction']
        )
        
        output_text = self.construct_output(
            demo_data['actions'],
            demo_data['grippers']
        )
        
        return input_text, output_text
    
    def construct_icl_prompt(self,
                            demonstrations: List[Dict],
                            test_data: Dict) -> str:
        """
        Construct full ICL prompt with demonstrations and test input.
        
        Format: x1 > y1, x2 > y2, ..., xn > yn, xTest >
        
        Args:
            demonstrations: List of demonstration dictionaries
            test_data: Test input dictionary with 'object_poses' and 'instruction'
            
        Returns:
            prompt: Full ICL prompt string
        """
        # Limit number of demonstrations
        demonstrations = demonstrations[:self.num_demonstrations]
        
        # Construct demonstration pairs
        demo_pairs = []
        for demo in demonstrations:
            input_text, output_text = self.construct_demonstration(demo)
            demo_pairs.append(f"{input_text} > {output_text}")
        
        # Construct test input
        test_input = self.construct_input(
            test_data['object_poses'],
            test_data['instruction']
        )
        
        # Combine everything
        prompt = ", ".join(demo_pairs)
        prompt += f", {test_input} >"
        
        return prompt
    
    def construct_icl_prompt_with_system(self,
                                        demonstrations: List[Dict],
                                        test_data: Dict) -> Tuple[str, str]:
        """
        Construct ICL prompt with system message.
        
        Args:
            demonstrations: List of demonstration dictionaries
            test_data: Test input dictionary
            
        Returns:
            system_message: System prompt
            user_prompt: User prompt with demonstrations
        """
        system_message = self._get_system_message()
        user_prompt = self.construct_icl_prompt(demonstrations, test_data)
        
        return system_message, user_prompt
    
    def _get_system_message(self) -> str:
        """
        Get system message for LLM.
        
        Returns:
            system_message: System prompt describing the task
        """
        system_message = """You are a robot action prediction system. Given object poses and a task instruction, predict the sequence of robot actions needed to complete the task.

Input format: {[object1]: [x, y, z, roll, pitch, yaw], [object2]: [x, y, z, roll, pitch, yaw], ..., [Task]: description}
Output format: {[x, y, z, roll, pitch, yaw, gripper], [x, y, z, roll, pitch, yaw, gripper], ...}

Where:
- x, y, z: Position bins (0-99)
- roll, pitch, yaw: Rotation bins (0-71)
- gripper: Binary state (0=open, 1=closed)

You will be given several example input-output pairs, followed by a test input. Predict the output for the test input based on the patterns you observe in the examples.

Respond ONLY with the predicted action sequence in the exact format shown, no explanations."""
        
        return system_message
    
    def parse_llm_response(self, response: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Parse LLM response to extract predicted actions.
        
        Args:
            response: LLM response string
            
        Returns:
            actions: List of discrete action poses
            grippers: List of discrete gripper states
        """
        # Extract the output portion (between curly braces)
        try:
            # Find content between {}
            start = response.find('{')
            end = response.rfind('}')
            
            if start == -1 or end == -1:
                raise ValueError("No valid output format found in response")
            
            content = response[start+1:end].strip()
            
            # Split by commas to get individual actions
            # But need to handle nested brackets carefully
            action_strings = self._split_actions(content)
            
            actions = []
            grippers = []
            
            for action_str in action_strings:
                action_str = action_str.strip()
                if not action_str:
                    continue
                
                # Parse discrete pose
                discrete_pose, discrete_gripper = self.discretizer.parse_discrete_pose_from_text(action_str)
                actions.append(discrete_pose)
                grippers.append(discrete_gripper)
            
            return actions, grippers
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response: {response}")
            return [], []
    
    def _split_actions(self, content: str) -> List[str]:
        """
        Split action string by commas, handling nested brackets.
        
        Args:
            content: String content between outer braces
            
        Returns:
            action_strings: List of action strings
        """
        actions = []
        current_action = ""
        bracket_depth = 0
        
        for char in content:
            if char == '[':
                bracket_depth += 1
                current_action += char
            elif char == ']':
                bracket_depth -= 1
                current_action += char
                
                # If we're back to depth 0, we've completed an action
                if bracket_depth == 0 and current_action.strip():
                    actions.append(current_action.strip())
                    current_action = ""
            elif char == ',' and bracket_depth == 0:
                # Skip commas at depth 0 (between actions)
                continue
            else:
                current_action += char
        
        # Add any remaining action
        if current_action.strip():
            actions.append(current_action.strip())
        
        return actions
    
    def format_demonstration_for_display(self, demo_data: Dict) -> str:
        """
        Format a demonstration for human-readable display.
        
        Args:
            demo_data: Demonstration dictionary
            
        Returns:
            formatted: Formatted string
        """
        input_text, output_text = self.construct_demonstration(demo_data)
        
        formatted = "="*60 + "\n"
        formatted += "DEMONSTRATION\n"
        formatted += "="*60 + "\n"
        formatted += f"INPUT:\n{input_text}\n\n"
        formatted += f"OUTPUT:\n{output_text}\n"
        formatted += "="*60 + "\n"
        
        return formatted


def test_icl_constructor():
    """Test ICL constructor with sample data."""
    from action_discretizer import ActionDiscretizer
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create discretizer and constructor
    discretizer = ActionDiscretizer(config)
    constructor = ICLConstructor(config, discretizer)
    
    # Create sample demonstrations
    demonstrations = []
    for i in range(3):
        # Sample object poses
        object_poses = {
            'tape': np.array([10 + i*5, 20, 30, 5, 10, 15]),
            'target': np.array([50, 60, 70, 10, 20, 30])
        }
        
        # Sample actions (keyframes)
        actions = [
            np.array([15, 25, 35, 8, 12, 18]),
            np.array([25, 35, 45, 10, 15, 20]),
            np.array([50, 60, 70, 10, 20, 30])
        ]
        grippers = [0, 1, 0]
        
        demo = {
            'object_poses': object_poses,
            'instruction': 'Pick up the tape and place it on the target',
            'actions': actions,
            'grippers': grippers
        }
        demonstrations.append(demo)
    
    # Create test input
    test_data = {
        'object_poses': {
            'tape': np.array([12, 22, 32, 6, 11, 16]),
            'target': np.array([48, 58, 68, 9, 19, 29])
        },
        'instruction': 'Pick up the tape and place it on the target'
    }
    
    # Construct ICL prompt
    system_msg, user_prompt = constructor.construct_icl_prompt_with_system(
        demonstrations, test_data
    )
    
    print("System Message:")
    print(system_msg)
    print("\n" + "="*60 + "\n")
    print("User Prompt:")
    print(user_prompt)
    
    # Test parsing (with dummy LLM response)
    dummy_response = "{[15, 25, 35, 8, 12, 18, 0], [25, 35, 45, 10, 15, 20, 1], [48, 58, 68, 9, 19, 29, 0]}"
    print("\n" + "="*60 + "\n")
    print("Test Parsing:")
    print(f"Dummy LLM response: {dummy_response}")
    
    parsed_actions, parsed_grippers = constructor.parse_llm_response(dummy_response)
    print(f"Parsed actions: {parsed_actions}")
    print(f"Parsed grippers: {parsed_grippers}")


if __name__ == "__main__":
    test_icl_constructor()
