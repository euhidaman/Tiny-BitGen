"""
BitMar GRPO Reasoning Module
Implements Tiny-R1 style chain-of-thought reasoning with GRPO optimization
Integrates robot selection reasoning directly into the BitMar model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import re
from transformers import AutoTokenizer
from fiber_fusion import BitNetLinear

logger = logging.getLogger(__name__)


class BitMarGRPOReasoningModule(nn.Module):
    """
    GRPO-based reasoning module for BitMar model
    Implements chain-of-thought reasoning with policy optimization
    Specialized for robot selection tasks with vision-language grounding
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        max_reasoning_steps: int = 5,
        reasoning_temperature: float = 0.7,
        grpo_config: Optional[Dict] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_reasoning_steps = max_reasoning_steps
        self.reasoning_temperature = reasoning_temperature

        # GRPO configuration
        self.grpo_config = grpo_config or {
            "learning_rate": 1e-6,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "reward_scaling": 1.0
        }

        # Reasoning chain components
        self.thought_encoder = BitNetLinear(hidden_dim, hidden_dim)
        self.thought_generator = BitNetLinear(hidden_dim, hidden_dim)
        self.thought_evaluator = BitNetLinear(hidden_dim, 1)  # Score thoughts
        self.thought_refiner = BitNetLinear(hidden_dim * 2, hidden_dim)

        # Chain-of-thought processing
        self.reasoning_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # GRPO components
        self.value_head = BitNetLinear(hidden_dim, 1)  # Value function
        # Policy for reasoning tokens
        self.policy_head = BitNetLinear(hidden_dim, vocab_size)
        self.action_head = BitNetLinear(
            hidden_dim, 6)  # Robot selection actions

        # Robot selection reasoning
        self.robot_embeddings = nn.Embedding(
            6, hidden_dim)  # 5 robots + no-robot
        self.robot_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Reasoning quality assessment
        self.reasoning_quality_scorer = BitNetLinear(
            hidden_dim, 4)  # Quality dimensions

        # Layer norms
        self.thought_norm = nn.LayerNorm(hidden_dim)
        self.reasoning_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Robot capability knowledge
        self.robot_capabilities = self._initialize_robot_knowledge()

        logger.info(
            f"BitMar GRPO Reasoning Module initialized with {max_reasoning_steps} reasoning steps")

    def _initialize_robot_knowledge(self) -> Dict[str, int]:
        """Initialize robot type to index mapping"""
        return {
            "drone": 0,
            "underwater_robot": 1,
            "humanoid": 2,
            "robot_with_wheels": 3,
            "robot_with_legs": 4,
            "no_robot": 5
        }

    def generate_reasoning_chain(
        self,
        context: torch.Tensor,  # [batch_size, hidden_dim]
        vision_features: Optional[torch.Tensor] = None,  # [batch_size, vision_dim] - now optional
        task_description: Optional[str] = None,
        return_intermediate: bool = True,
        reasoning_mode: str = "vision_language"  # "vision_language" or "text_only"
    ) -> Dict[str, torch.Tensor]:
        """
        Generate chain-of-thought reasoning for robot selection
        Supports both vision-grounded (COCO) and text-only (robot selection) reasoning

        Args:
            context: Fused vision-language context from FIBER (for COCO) or text context (for robot tasks)
            vision_features: Raw vision features for grounding (None for text-only robot tasks)
            task_description: Text description of the task
            return_intermediate: Whether to return intermediate reasoning steps
            reasoning_mode: "vision_language" for COCO training, "text_only" for robot selection

        Returns:
            Dict containing reasoning outputs and intermediate states
        """
        batch_size = context.size(0)
        device = context.device

        # Initialize reasoning state
        reasoning_states = []
        thought_scores = []
        robot_selections = []

        # Initial reasoning state - enhanced for text-only mode
        if reasoning_mode == "text_only":
            # For robot selection: use learned visual knowledge to infer environment
            current_state = self._infer_environment_from_text(context, task_description)
        else:
            # For COCO: use standard vision-language context
            current_state = self.thought_encoder(context)

        reasoning_states.append(current_state)

        # LSTM hidden states for reasoning continuity
        h_0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        c_0 = torch.zeros(2, batch_size, self.hidden_dim, device=device)
        lstm_hidden = (h_0, c_0)

        # Reasoning chain generation with mode-specific steps
        for step in range(self.max_reasoning_steps):
            # Generate thought for current step with mode-aware reasoning
            thought_input = current_state.unsqueeze(1)  # [batch, 1, hidden]
            thought_output, lstm_hidden = self.reasoning_lstm(
                thought_input, lstm_hidden)
            thought = thought_output.squeeze(1)  # [batch, hidden]

            # Apply thought generation transformation
            if reasoning_mode == "text_only":
                generated_thought = self._generate_environment_aware_thought(
                    thought, step, task_description)
            else:
                generated_thought = self.thought_generator(thought)

            generated_thought = self.thought_norm(generated_thought)

            # Evaluate thought quality
            thought_score = self.thought_evaluator(generated_thought)
            thought_scores.append(thought_score)

            # Robot selection reasoning - enhanced for both modes
            robot_reasoning = self._reason_about_robots(
                generated_thought,
                vision_features,
                step,
                reasoning_mode,
                task_description
            )
            robot_selections.append(robot_reasoning)

            # Refine state for next iteration
            if step < self.max_reasoning_steps - 1:
                refined_input = torch.cat(
                    [current_state, generated_thought], dim=-1)
                current_state = self.thought_refiner(refined_input)
                current_state = self.reasoning_norm(current_state)
                reasoning_states.append(current_state)
            else:
                reasoning_states.append(generated_thought)

        # Final reasoning state
        final_reasoning_state = reasoning_states[-1]
        final_reasoning_state = self.output_norm(final_reasoning_state)

        # Aggregate robot selections across reasoning steps
        final_robot_selection = self._aggregate_robot_selections(
            robot_selections)

        # Compute reasoning quality
        reasoning_quality = self.reasoning_quality_scorer(
            final_reasoning_state)

        result = {
            "final_reasoning_state": final_reasoning_state,
            "robot_selection": final_robot_selection,
            "reasoning_quality": reasoning_quality,
            # [batch, steps, 1]
            "thought_scores": torch.stack(thought_scores, dim=1),
        }

        if return_intermediate:
            result.update({
                # [batch, steps+1, hidden]
                "reasoning_states": torch.stack(reasoning_states, dim=1),
                "robot_selections_per_step": robot_selections,
            })

        return result

    def _reason_about_robots(
        self,
        thought: torch.Tensor,  # [batch_size, hidden_dim]
        vision_features: Optional[torch.Tensor],
        step: int,
        reasoning_mode: str = "vision_language",
        task_description: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Reason about robot selection for current thought
        """
        batch_size = thought.size(0)
        device = thought.device

        # Create robot embeddings
        robot_indices = torch.arange(
            6, device=device).unsqueeze(0).expand(batch_size, -1)
        robot_embeds = self.robot_embeddings(
            robot_indices)  # [batch, 6, hidden]

        # Attention between thought and robot capabilities
        thought_query = thought.unsqueeze(1)  # [batch, 1, hidden]

        attended_robots, robot_attention_weights = self.robot_attention(
            query=thought_query,
            key=robot_embeds,
            value=robot_embeds
        )  # [batch, 1, hidden], [batch, 1, 6]

        attended_robots = attended_robots.squeeze(1)  # [batch, hidden]
        robot_attention_weights = robot_attention_weights.squeeze(
            1)  # [batch, 6]

        # Robot selection policy
        robot_logits = self.action_head(attended_robots)  # [batch, 6]
        robot_probs = F.softmax(
            robot_logits / self.reasoning_temperature, dim=-1)

        # Value estimation for this robot reasoning step
        step_value = self.value_head(attended_robots)  # [batch, 1]

        return {
            "robot_logits": robot_logits,
            "robot_probs": robot_probs,
            "robot_attention": robot_attention_weights,
            "step_value": step_value,
            "attended_robots": attended_robots
        }

    def _aggregate_robot_selections(
        self,
        robot_selections: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate robot selections across reasoning steps
        """
        # Stack all robot probabilities
        all_robot_probs = torch.stack(
            [sel["robot_probs"] for sel in robot_selections], dim=1)
        all_robot_logits = torch.stack(
            [sel["robot_logits"] for sel in robot_selections], dim=1)

        # Weighted average based on step values
        step_values = torch.stack([sel["step_value"]
                                  for sel in robot_selections], dim=1)
        step_weights = F.softmax(
            step_values.squeeze(-1), dim=1)  # [batch, steps]

        # Weighted aggregation
        final_robot_probs = torch.sum(
            all_robot_probs * step_weights.unsqueeze(-1), dim=1
        )  # [batch, 6]

        final_robot_logits = torch.sum(
            all_robot_logits * step_weights.unsqueeze(-1), dim=1
        )  # [batch, 6]

        return {
            "final_robot_probs": final_robot_probs,
            "final_robot_logits": final_robot_logits,
            "step_weights": step_weights,
            "all_step_probs": all_robot_probs
        }

    def compute_grpo_loss(
        self,
        reasoning_output: Dict[str, torch.Tensor],
        robot_rewards: torch.Tensor,  # [batch_size]
        reasoning_rewards: torch.Tensor,  # [batch_size]
        baseline_value: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss for reasoning optimization

        Args:
            reasoning_output: Output from generate_reasoning_chain
            robot_rewards: Rewards for robot selection accuracy
            reasoning_rewards: Rewards for reasoning quality
            baseline_value: Baseline value for advantage computation

        Returns:
            Dict containing loss components
        """
        device = robot_rewards.device
        batch_size = robot_rewards.size(0)

        # Extract components
        robot_selection = reasoning_output["robot_selection"]
        # [batch, steps, 1]
        thought_scores = reasoning_output["thought_scores"]
        robot_selections_per_step = reasoning_output.get(
            "robot_selections_per_step", [])

        # Compute total reward
        total_reward = robot_rewards + reasoning_rewards

        # Value function loss
        if baseline_value is None:
            # Use average step values as baseline
            step_values = torch.stack([
                sel["step_value"].squeeze(-1) for sel in robot_selections_per_step
            ], dim=1)  # [batch, steps]
            baseline_value = step_values.mean(dim=1)  # [batch]

        value_loss = F.mse_loss(baseline_value, total_reward)

        # Policy gradient loss
        advantages = total_reward - baseline_value.detach()  # [batch]

        # Robot selection policy loss
        robot_log_probs = F.log_softmax(
            robot_selection["final_robot_logits"], dim=-1)
        robot_policy_loss = 0

        # For each step, compute policy gradient
        for step_idx, step_selection in enumerate(robot_selections_per_step):
            step_log_probs = F.log_softmax(
                step_selection["robot_logits"], dim=-1)
            step_robot_probs = step_selection["robot_probs"]

            # Sample actions based on probabilities
            robot_actions = torch.multinomial(step_robot_probs, 1).squeeze(-1)
            selected_log_probs = step_log_probs.gather(
                1, robot_actions.unsqueeze(-1)).squeeze(-1)

            # Weighted by step importance
            step_weights = robot_selection["step_weights"][:, step_idx]
            step_policy_loss = -selected_log_probs * advantages * step_weights
            robot_policy_loss += step_policy_loss.mean()

        # Reasoning quality loss (auxiliary)
        reasoning_quality = reasoning_output["reasoning_quality"]  # [batch, 4]
        quality_targets = self._compute_quality_targets(
            reasoning_rewards)  # [batch, 4]
        quality_loss = F.mse_loss(reasoning_quality, quality_targets)

        # Entropy bonus for exploration
        robot_entropy = -torch.sum(
            robot_selection["final_robot_probs"] *
            torch.log(robot_selection["final_robot_probs"] + 1e-8),
            dim=-1
        ).mean()

        # Total loss
        total_loss = (
            robot_policy_loss +
            self.grpo_config["value_loss_coef"] * value_loss +
            0.1 * quality_loss -
            self.grpo_config["entropy_coef"] * robot_entropy
        )

        return {
            "total_loss": total_loss,
            "policy_loss": robot_policy_loss,
            "value_loss": value_loss,
            "quality_loss": quality_loss,
            "entropy": robot_entropy,
            "advantages": advantages
        }

    def _compute_quality_targets(self, reasoning_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute quality targets for reasoning assessment
        """
        batch_size = reasoning_rewards.size(0)
        device = reasoning_rewards.device

        # Quality dimensions: coherence, relevance, completeness, accuracy
        quality_targets = torch.zeros(batch_size, 4, device=device)

        # Simple mapping from rewards to quality dimensions
        quality_targets[:, 0] = torch.clamp(
            reasoning_rewards * 0.8, 0, 1)  # coherence
        quality_targets[:, 1] = torch.clamp(
            reasoning_rewards * 0.9, 0, 1)  # relevance
        quality_targets[:, 2] = torch.clamp(
            reasoning_rewards * 0.7, 0, 1)  # completeness
        quality_targets[:, 3] = torch.clamp(
            reasoning_rewards, 0, 1)         # accuracy

        return quality_targets

    def extract_robot_selection_text(
        self,
        robot_probs: torch.Tensor,  # [batch_size, 6]
        threshold: float = 0.3
    ) -> List[str]:
        """
        Convert robot probabilities to human-readable selections
        """
        robot_names = [
            "Drone", "Underwater Robot", "Humanoid",
            "Robot with Wheels", "Robot with Legs", "No Robot"
        ]

        batch_size = robot_probs.size(0)
        selections = []

        for batch_idx in range(batch_size):
            selected_robots = []
            probs = robot_probs[batch_idx]

            for robot_idx, prob in enumerate(probs):
                if prob > threshold and robot_idx < 5:  # Exclude "No Robot"
                    selected_robots.append(robot_names[robot_idx])

            if not selected_robots:
                # If no robot meets threshold, select the highest probability one
                max_idx = torch.argmax(probs[:5])  # Exclude "No Robot"
                selected_robots.append(robot_names[max_idx])

            selections.append(", ".join(selected_robots))

        return selections

    def _infer_environment_from_text(
        self,
        context: torch.Tensor,
        task_description: Optional[str] = None
    ) -> torch.Tensor:
        """
        Use learned visual knowledge to infer environment from text description
        This leverages the model's COCO-trained visual understanding
        """
        # Start with text context encoding
        inferred_context = self.thought_encoder(context)

        # Add environment inference based on task keywords
        if task_description:
            # Environment inference based on learned visual patterns
            environment_features = self._extract_environment_cues(task_description)
            # Combine text context with inferred visual environment
            inferred_context = inferred_context + environment_features

        return inferred_context

    def _extract_environment_cues(self, task_description: str) -> torch.Tensor:
        """
        Extract environment cues from task description using learned visual knowledge
        Maps text descriptions to visual feature spaces learned from COCO
        """
        device = next(self.parameters()).device
        batch_size = 1  # Assuming single task for now

        # Create environment feature vector based on task keywords
        environment_vector = torch.zeros(self.hidden_dim, device=device)

        task_lower = task_description.lower()

        # Indoor environments (from COCO indoor scenes)
        if any(word in task_lower for word in ['indoor', 'mall', 'warehouse', 'building', 'room', 'stairs']):
            environment_vector[:64] += 0.5  # Indoor feature space

        # Outdoor environments (from COCO outdoor scenes)
        if any(word in task_lower for word in ['outdoor', 'mountain', 'desert', 'field', 'terrain', 'path']):
            environment_vector[64:128] += 0.5  # Outdoor feature space

        # Water environments (inferred from COCO water scenes)
        if any(word in task_lower for word in ['underwater', 'water', 'ocean', 'sea', 'marine', 'pipes']):
            environment_vector[128:192] += 0.7  # Water feature space

        # Aerial environments (from COCO sky/building scenes)
        if any(word in task_lower for word in ['aerial', 'high-rise', 'above', 'building', 'exterior']):
            environment_vector[192:256] += 0.6  # Aerial feature space

        # Crowded/human environments (from COCO people scenes)
        if any(word in task_lower for word in ['crowded', 'people', 'pedestrians', 'urban', 'human']):
            environment_vector[256:320] += 0.5  # Social feature space

        # Rough terrain (inferred from COCO outdoor/nature scenes)
        if any(word in task_lower for word in ['rough', 'rocky', 'uneven', 'obstacles', 'damaged']):
            environment_vector[320:384] += 0.6  # Rough terrain feature space

        return environment_vector.unsqueeze(0)  # Add batch dimension

    def _generate_environment_aware_thought(
        self,
        thought: torch.Tensor,
        step: int,
        task_description: Optional[str] = None
    ) -> torch.Tensor:
        """
        Generate environment-aware thoughts for text-only robot reasoning
        Each step focuses on different aspects of environment-robot matching
        """
        # Base thought transformation
        base_thought = self.thought_generator(thought)

        if task_description is None:
            return base_thought

        # Step-specific reasoning enhancement
        task_lower = task_description.lower()

        if step == 0:  # Environment Analysis
            # Focus on understanding the environment from text
            env_modifier = self._get_environment_analysis_modifier(task_lower)

        elif step == 1:  # Terrain/Space Assessment
            # Focus on physical constraints and space requirements
            env_modifier = self._get_terrain_assessment_modifier(task_lower)

        elif step == 2:  # Robot Capability Matching
            # Focus on matching robot capabilities to environment needs
            env_modifier = self._get_capability_matching_modifier(task_lower)

        elif step == 3:  # Constraint Evaluation
            # Focus on limitations and constraints
            env_modifier = self._get_constraint_evaluation_modifier(task_lower)

        else:  # step == 4: Final Selection & Confidence
            # Focus on final decision and confidence assessment
            env_modifier = self._get_final_selection_modifier(task_lower)

        # Apply step-specific environment reasoning
        enhanced_thought = base_thought + env_modifier

        return enhanced_thought

    def _get_environment_analysis_modifier(self, task_lower: str) -> torch.Tensor:
        """Step 1: Environment Analysis - 'I observe/infer this type of environment'"""
        device = next(self.parameters()).device
        modifier = torch.zeros(self.hidden_dim, device=device)

        # Boost features that help identify environment type
        if 'indoor' in task_lower or 'mall' in task_lower or 'warehouse' in task_lower:
            modifier[:128] += 0.3  # Indoor recognition boost
        elif 'outdoor' in task_lower or 'terrain' in task_lower or 'mountain' in task_lower:
            modifier[128:256] += 0.3  # Outdoor recognition boost
        elif 'underwater' in task_lower or 'water' in task_lower:
            modifier[256:384] += 0.4  # Water recognition boost
        elif 'aerial' in task_lower or 'high-rise' in task_lower:
            modifier[384:512] += 0.3  # Aerial recognition boost

        return modifier.unsqueeze(0)

    def _get_terrain_assessment_modifier(self, task_lower: str) -> torch.Tensor:
        """Step 2: Terrain Assessment - 'This requires specific navigation capabilities'"""
        device = next(self.parameters()).device
        modifier = torch.zeros(self.hidden_dim, device=device)

        # Boost features that assess physical requirements
        if any(word in task_lower for word in ['rough', 'rocky', 'uneven', 'obstacles']):
            modifier[100:200] += 0.4  # Rough terrain assessment
        elif any(word in task_lower for word in ['flat', 'smooth', 'warehouse']):
            modifier[200:300] += 0.3  # Smooth surface assessment
        elif any(word in task_lower for word in ['crowded', 'people', 'pedestrians']):
            modifier[300:400] += 0.3  # Social navigation assessment
        elif any(word in task_lower for word in ['stairs', 'vertical', 'climbing']):
            modifier[400:500] += 0.4  # Vertical movement assessment

        return modifier.unsqueeze(0)

    def _get_capability_matching_modifier(self, task_lower: str) -> torch.Tensor:
        """Step 3: Robot Capability Matching - 'Robot X has the right capabilities'"""
        device = next(self.parameters()).device
        modifier = torch.zeros(self.hidden_dim, device=device)

        # Boost features that match capabilities to requirements
        if any(word in task_lower for word in ['aerial', 'above', 'survey', 'inspect.*above']):
            modifier[50:150] += 0.5  # Drone capability matching
        elif any(word in task_lower for word in ['underwater', 'marine', 'sea']):
            modifier[150:250] += 0.6  # Underwater robot matching
        elif any(word in task_lower for word in ['human', 'interaction', 'crowded', 'stairs']):
            modifier[250:350] += 0.4  # Humanoid matching
        elif any(word in task_lower for word in ['fast', 'warehouse', 'flat', 'transport']):
            modifier[350:450] += 0.4  # Wheeled robot matching
        elif any(word in task_lower for word in ['rough', 'terrain', 'mountain', 'uneven']):
            modifier[450:550] += 0.5  # Legged robot matching

        return modifier.unsqueeze(0)

    def _get_constraint_evaluation_modifier(self, task_lower: str) -> torch.Tensor:
        """Step 4: Constraint Evaluation - 'Other robots would fail because...'"""
        device = next(self.parameters()).device
        modifier = torch.zeros(self.hidden_dim, device=device)

        # Boost features that evaluate limitations
        if any(word in task_lower for word in ['water', 'underwater']):
            modifier[0:100] += 0.5  # Water constraint evaluation (eliminates most robots)
        elif any(word in task_lower for word in ['rough', 'uneven', 'rocky']):
            modifier[100:200] += 0.4  # Terrain constraint (eliminates wheels)
        elif any(word in task_lower for word in ['indoor', 'crowded', 'people']):
            modifier[200:300] += 0.3  # Social constraint (favors humanoid)
        elif any(word in task_lower for word in ['high', 'aerial', 'above']):
            modifier[300:400] += 0.4  # Height constraint (favors drone)

        return modifier.unsqueeze(0)

    def _get_final_selection_modifier(self, task_lower: str) -> torch.Tensor:
        """Step 5: Final Selection - 'Therefore, Robot X is optimal with confidence Y'"""
        device = next(self.parameters()).device
        modifier = torch.zeros(self.hidden_dim, device=device)

        # Boost features that support confident final selection
        modifier[600:] += 0.2  # General confidence boost for final decision

        # Task-specific confidence boosts
        if any(word in task_lower for word in ['underwater', 'marine']):
            modifier[600:650] += 0.4  # High confidence for underwater tasks
        elif any(word in task_lower for word in ['aerial', 'survey', 'above']):
            modifier[650:700] += 0.4  # High confidence for aerial tasks

        return modifier.unsqueeze(0)

    def forward(
        self,
        context: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        robot_labels: Optional[torch.Tensor] = None,
        reasoning_rewards: Optional[torch.Tensor] = None,
        robot_rewards: Optional[torch.Tensor] = None,
        return_loss: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GRPO reasoning module

        Args:
            context: Fused context from BitMar cross-modal fusion
            vision_features: Raw vision features for grounding
            robot_labels: Ground truth robot selections for training
            reasoning_rewards: Rewards for reasoning quality
            robot_rewards: Rewards for robot selection accuracy
            return_loss: Whether to compute and return GRPO loss

        Returns:
            Dict containing reasoning outputs and optionally loss
        """
        # Generate reasoning chain
        reasoning_output = self.generate_reasoning_chain(
            context=context,
            vision_features=vision_features,
            return_intermediate=True
        )

        result = {
            "reasoning_state": reasoning_output["final_reasoning_state"],
            "robot_selection": reasoning_output["robot_selection"]["final_robot_probs"],
            "robot_logits": reasoning_output["robot_selection"]["final_robot_logits"],
            "reasoning_quality": reasoning_output["reasoning_quality"],
            "thought_scores": reasoning_output["thought_scores"]
        }

        # Compute loss if training
        if return_loss and robot_rewards is not None and reasoning_rewards is not None:
            loss_dict = self.compute_grpo_loss(
                reasoning_output=reasoning_output,
                robot_rewards=robot_rewards,
                reasoning_rewards=reasoning_rewards
            )
            result.update(loss_dict)

        return result


def create_grpo_reasoning_module(config: Dict) -> BitMarGRPOReasoningModule:
    """
    Factory function to create GRPO reasoning module from config
    """
    grpo_config = config.get('grpo_reasoning', {})

    return BitMarGRPOReasoningModule(
        hidden_dim=config.get('text_encoder_dim', 128),
        vocab_size=config.get('vocab_size', 50257),
        max_reasoning_steps=grpo_config.get('max_reasoning_steps', 5),
        reasoning_temperature=grpo_config.get('reasoning_temperature', 0.7),
        grpo_config=grpo_config.get('training', {})
    )


if __name__ == "__main__":
    # Test the GRPO reasoning module
    config = {
        'text_encoder_dim': 128,
        'vocab_size': 50257,
        'grpo_reasoning': {
            'max_reasoning_steps': 3,
            'reasoning_temperature': 0.7
        }
    }

    module = create_grpo_reasoning_module(config)

    # Test forward pass
    batch_size = 2
    context = torch.randn(batch_size, 128)
    vision_features = torch.randn(batch_size, 768)

    with torch.no_grad():
        output = module(context, vision_features)

    print("GRPO Reasoning Module Test:")
    print(f"Reasoning state shape: {output['reasoning_state'].shape}")
    print(f"Robot selection shape: {output['robot_selection'].shape}")
    print(
        f"Robot selections: {module.extract_robot_selection_text(output['robot_selection'])}")
