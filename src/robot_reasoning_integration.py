"""
BitMar Robot Reasoning Integration
Integrates GRPO-trained robot selection reasoning with the existing BitMar model
"""

import torch
import torch.nn as nn
import json
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from src.model import BitMarModel
from src.robot_grpo_training import RobotSelectionRewardFunctions, ROBOT_SELECTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class BitMarRobotReasoner(nn.Module):
    """
    BitMar model enhanced with robot selection reasoning capabilities
    Combines vision-language understanding with structured robot selection reasoning
    """
    
    def __init__(
        self,
        base_bitmar_model: BitMarModel,
        grpo_model_path: Optional[str] = None,
        robot_reasoning_config: Optional[Dict] = None
    ):
        super().__init__()
        self.base_model = base_bitmar_model
        self.config = base_bitmar_model.config
        
        # Robot reasoning configuration
        self.robot_reasoning_config = robot_reasoning_config or {
            "max_reasoning_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "reasoning_weight": 0.3,
            "vision_reasoning_fusion": True
        }
        
        # Load GRPO-trained reasoning model if provided
        if grpo_model_path:
            self.reasoning_model = AutoModelForCausalLM.from_pretrained(grpo_model_path)
            self.reasoning_tokenizer = AutoTokenizer.from_pretrained(grpo_model_path)
            if self.reasoning_tokenizer.pad_token is None:
                self.reasoning_tokenizer.pad_token = self.reasoning_tokenizer.eos_token
        else:
            self.reasoning_model = None
            self.reasoning_tokenizer = None
        
        # Robot capability database
        self.robot_capabilities = self._load_robot_capabilities()
        
        # Reasoning integration layer
        if self.reasoning_model:
            self.reasoning_fusion = nn.Linear(
                self.config['text_encoder_dim'] + self.reasoning_model.config.hidden_size,
                self.config['text_encoder_dim']
            )
        
        logger.info("BitMar Robot Reasoner initialized with GRPO reasoning capabilities")
    
    def _load_robot_capabilities(self) -> Dict[str, Dict]:
        """Load robot capabilities database"""
        return {
            "drone": {
                "capabilities": ["fastest", "aerial navigation", "surveillance", "lightweight transport", "aerial inspection"],
                "limitations": ["limited payload", "loud", "weather dependent", "battery life"],
                "environments": ["outdoor", "large indoor spaces", "hard to reach areas"]
            },
            "humanoid": {
                "capabilities": ["manipulation", "walking", "human interaction", "complex tasks", "tool use"],
                "limitations": ["slow movement", "balance issues", "high power consumption"],
                "environments": ["indoor", "human environments", "stairs", "complex terrain"]
            },
            "robot_with_legs": {
                "capabilities": ["rough terrain navigation", "stability", "load carrying", "inspection"],
                "limitations": ["limited manipulation", "height restrictions"],
                "environments": ["outdoor", "uneven terrain", "stairs", "industrial sites", "search and rescue"]
            },
            "robot_with_wheels": {
                "capabilities": ["fast movement", "good payload", "stable platform", "efficient"],
                "limitations": ["flat surfaces only", "limited climbing", "obstacle avoidance"],
                "environments": ["indoor", "warehouse", "flat outdoor areas", "roads"]
            },
            "underwater_robot": {
                "capabilities": ["underwater navigation", "deep sea exploration", "marine inspection", "underwater manipulation"],
                "limitations": ["water environments only", "communication limitations", "pressure constraints"],
                "environments": ["underwater", "marine", "pools", "pipes", "ocean exploration"]
            }
        }
    
    def format_robot_selection_prompt(
        self,
        task_description: str,
        include_vision_context: bool = True,
        vision_features: Optional[torch.Tensor] = None
    ) -> str:
        """
        Format a robot selection prompt with task description and robot capabilities
        """
        # Build robot capabilities description
        robot_desc = "Robots: Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs. What is the most suited robot for the task? It can be more than one robot if they are both equally suited for the task.\n"
        
        for robot_name, info in self.robot_capabilities.items():
            display_name = robot_name.replace("_", " ").title()
            if display_name == "Robot With Legs":
                display_name = "Robot with Legs"
            elif display_name == "Robot With Wheels":
                display_name = "Robot with Wheels"
            
            robot_desc += f"{display_name}:\n"
            robot_desc += f"capabilities: {', '.join(info['capabilities'])},\n"
            robot_desc += f"limitations: {', '.join(info['limitations'])},\n"
            robot_desc += f"environments: {', '.join(info['environments'])}\n"
        
        # Add vision context if available
        vision_context = ""
        if include_vision_context and vision_features is not None:
            vision_context = "\n[Visual context: Image features have been analyzed and will inform the robot selection decision]"
        
        prompt = f"{robot_desc}{vision_context}"
        return prompt
    
    def generate_robot_reasoning(
        self,
        task_description: str,
        vision_features: Optional[torch.Tensor] = None,
        return_reasoning_steps: bool = True
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Generate structured reasoning for robot selection
        """
        if self.reasoning_model is None:
            logger.warning("No GRPO reasoning model loaded, using base model")
            return self._fallback_reasoning(task_description)
        
        # Format prompt
        robot_prompt = self.format_robot_selection_prompt(
            task_description, 
            include_vision_context=(vision_features is not None),
            vision_features=vision_features
        )
        
        # Create conversation format
        messages = [
            {"role": "system", "content": ROBOT_SELECTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{robot_prompt}\n\nTask: {task_description}"}
        ]
        
        # Tokenize
        input_text = self.reasoning_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.reasoning_tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        
        # Move to device
        device = next(self.reasoning_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate reasoning
        with torch.no_grad():
            outputs = self.reasoning_model.generate(
                **inputs,
                max_new_tokens=self.robot_reasoning_config["max_reasoning_length"],
                temperature=self.robot_reasoning_config["temperature"],
                top_p=self.robot_reasoning_config["top_p"],
                do_sample=True,
                pad_token_id=self.reasoning_tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.reasoning_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Parse reasoning and answer
        reasoning_result = self._parse_reasoning_response(response)
        
        if return_reasoning_steps:
            reasoning_result["reasoning_steps"] = self._extract_reasoning_steps(
                reasoning_result.get("reasoning", "")
            )
        
        return reasoning_result
    
    def _parse_reasoning_response(self, response: str) -> Dict[str, str]:
        """Parse structured reasoning response"""
        result = {"reasoning": "", "selected_robots": "", "raw_response": response}
        
        try:
            # Extract reasoning
            if "<reasoning>" in response and "</reasoning>" in response:
                reasoning = response.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                result["reasoning"] = reasoning
            
            # Extract answer
            if "<answer>" in response and "</answer>" in response:
                answer = response.split("<answer>")[1].split("</answer>")[0].strip()
                # Clean up answer format
                answer = answer.replace("Selected robot(s):", "").strip()
                result["selected_robots"] = answer
        except Exception as e:
            logger.warning(f"Failed to parse reasoning response: {e}")
            result["selected_robots"] = "Unable to determine"
        
        return result
    
    def _extract_reasoning_steps(self, reasoning_text: str) -> List[str]:
        """Extract individual reasoning steps from reasoning text"""
        if not reasoning_text:
            return []
        
        # Split by sentences and filter meaningful steps
        sentences = [s.strip() for s in reasoning_text.split('.') if s.strip()]
        
        # Filter for reasoning steps (sentences that mention analysis, evaluation, etc.)
        reasoning_keywords = [
            "analyze", "consider", "evaluate", "assess", "determine", "require", 
            "suitable", "capability", "limitation", "environment", "task"
        ]
        
        steps = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in reasoning_keywords):
                steps.append(sentence + ".")
        
        return steps[:5]  # Limit to top 5 reasoning steps
    
    def _fallback_reasoning(self, task_description: str) -> Dict[str, str]:
        """Fallback reasoning when no GRPO model is available"""
        return {
            "reasoning": f"Analyzing task: {task_description}. Evaluating robot capabilities...",
            "selected_robots": "Multiple robots may be suitable",
            "reasoning_steps": ["Task analysis required", "Robot evaluation needed"],
            "raw_response": "Fallback reasoning - no GRPO model loaded"
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "train",
        include_robot_reasoning: bool = False,
        task_description: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional robot reasoning integration
        """
        # Base BitMar forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            vision_features=vision_features,
            labels=labels,
            mode=mode,
            **kwargs
        )
        
        # Add robot reasoning if requested
        if include_robot_reasoning and task_description:
            try:
                # Generate robot reasoning
                reasoning_result = self.generate_robot_reasoning(
                    task_description=task_description,
                    vision_features=vision_features,
                    return_reasoning_steps=True
                )
                
                # Add reasoning to outputs
                base_outputs.update({
                    "robot_reasoning": reasoning_result["reasoning"],
                    "selected_robots": reasoning_result["selected_robots"],
                    "reasoning_steps": reasoning_result.get("reasoning_steps", []),
                    "reasoning_confidence": self._compute_reasoning_confidence(reasoning_result)
                })
                
            except Exception as e:
                logger.warning(f"Robot reasoning failed: {e}")
                base_outputs.update({
                    "robot_reasoning": "Reasoning failed",
                    "selected_robots": "Unknown",
                    "reasoning_steps": [],
                    "reasoning_confidence": 0.0
                })
        
        return base_outputs
    
    def _compute_reasoning_confidence(self, reasoning_result: Dict) -> float:
        """Compute confidence score for reasoning quality"""
        confidence = 0.0
        
        # Check if reasoning exists and has good structure
        reasoning = reasoning_result.get("reasoning", "")
        if reasoning and len(reasoning.split()) > 10:
            confidence += 0.4
        
        # Check if specific robots are selected
        selected = reasoning_result.get("selected_robots", "")
        if selected and selected.lower() not in ["unknown", "unable to determine", ""]:
            confidence += 0.4
        
        # Check for reasoning keywords
        if reasoning:
            reasoning_keywords = ["analyze", "evaluate", "suitable", "capability", "task"]
            keyword_count = sum(1 for kw in reasoning_keywords if kw in reasoning.lower())
            confidence += min(keyword_count * 0.05, 0.2)
        
        return min(confidence, 1.0)


def create_robot_reasoner(
    base_model_config_path: str,
    grpo_model_path: Optional[str] = None,
    bitmar_checkpoint_path: Optional[str] = None
) -> BitMarRobotReasoner:
    """
    Factory function to create BitMar Robot Reasoner
    """
    import yaml
    
    # Load BitMar configuration
    with open(base_model_config_path, 'r') as f:
        config = yaml.safe_load(f)['model']
    
    # Initialize base BitMar model
    if bitmar_checkpoint_path:
        # Load from checkpoint
        base_model = BitMarModel.from_pretrained(bitmar_checkpoint_path)
    else:
        # Create new model
        base_model = BitMarModel(config)
    
    # Create robot reasoner
    robot_reasoner = BitMarRobotReasoner(
        base_bitmar_model=base_model,
        grpo_model_path=grpo_model_path
    )
    
    return robot_reasoner


if __name__ == "__main__":
    # Example usage
    reasoner = create_robot_reasoner(
        base_model_config_path="./configs/bitmar_coco.yaml",
        grpo_model_path="./bitmar_robot_reasoning",  # Path to GRPO-trained model
        bitmar_checkpoint_path="./checkpoints_coco/best_model"
    )
    
    # Test robot reasoning
    test_task = "Inspect underwater pipelines for damage and leaks"
    reasoning_result = reasoner.generate_robot_reasoning(test_task)
    
    print(f"Task: {test_task}")
    print(f"Reasoning: {reasoning_result['reasoning']}")
    print(f"Selected Robots: {reasoning_result['selected_robots']}")
    print(f"Reasoning Steps: {reasoning_result.get('reasoning_steps', [])}")