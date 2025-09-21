# BitMar: Vision-Language Episodic Memory Transformer with GRPO Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

BitMar is a **Vision-Language Episodic Memory Transformer** with integrated GRPO reasoning capabilities. It combines BitNet-quantized text processing, DiNOv2 vision encoding, episodic memory mechanisms, and Tiny-R1 style chain-of-thought reasoning for robot selection tasks.

## 🚀 Quick Start

Follow these commands sequentially to set up and run the complete BitMar system:

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/euhidaman/Tiny-BitGen.git
cd Tiny-BitGen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

#### COCO Dataset (for vision-language training)
```bash
# Download COCO supplement data
python download_coco_supplement.py --dataset_dir ./data/coco

# This will download:
# - COCO images and captions
# - Aligned pairs for vision-language training
# - Preprocessed vision features
```

#### Robot Selection Dataset (for reasoning training)
```bash
# Robot selection datasets should be placed at:
# D:/BabyLM/robot_selection_data/data/Single-Robot-Selection/single_robot_selection_dataset.json
# D:/BabyLM/robot_selection_data/data/Multi-Robot-Selection/multi_robot_selection_dataset.json

# If datasets are elsewhere, update paths in src/robot_selection_dataset.py
```

### 3. Configure Training
```bash
# Option 1: Use default config directly (recommended)
python train.py --config configs/bitmar_coco.yaml --device cuda:0

# Option 2: Create custom config (only if you need modifications)
cp configs/bitmar_coco.yaml configs/my_config.yaml
# Edit configs/my_config.yaml with your custom settings:
# - Dataset paths
# - Model dimensions  
# - Training parameters
# - Wandb project name (optional)
# - Hugging Face hub settings (optional)
```

### 4. Model Training
```bash
# Start hybrid COCO + Robot reasoning training
python train.py --config configs/my_config.yaml --device cuda:0

# With specific options:
python train.py \
    --config configs/my_config.yaml \
    --device cuda:0 \
    --enable_fast_eval \
    --save_every_n_steps 1000

# Monitor training with Weights & Biases (if configured)
# View logs: tail -f training_coco.log
```

### 5. Model Evaluation
```bash
# Evaluate on robot selection tasks
python -m src.robot_selection_dataset \
    --single_robot_path "path/to/single_robot_selection_dataset.json" \
    --multi_robot_path "path/to/multi_robot_selection_dataset.json"

# Test Chain-of-Thought reasoning
python -m src.grpo_reasoning_module

# Run comprehensive evaluation (if evaluation pipeline available)
python evaluate_model.py --checkpoint_path ./outputs/checkpoints/latest_checkpoint.pt
```

### 6. Inference and Usage
```bash
# Load trained model for inference
python -c "
from src.model import create_bitmar_model
import torch

# Load model
config = {'model': {...}}  # Your config
model = create_bitmar_model(config['model'])
checkpoint = torch.load('./outputs/checkpoints/latest_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Example robot selection reasoning
task = 'Navigate through underwater cave system'
# ... reasoning implementation
"
```

### 7. Optional: Hugging Face Integration
```bash
# If Hugging Face hub is configured, models are automatically uploaded
# Manual upload:
python -c "
from train import COCOTrainer
trainer = COCOTrainer('configs/my_config.yaml')
trainer.upload_checkpoint_to_hf(epoch=0, final=True)
"
```

### 8. Monitoring and Analysis
```bash
# View attention patterns
ls ./outputs/attention_analysis/

# Check memory usage patterns
ls ./outputs/memory_logs/

# Monitor carbon emissions (if enabled)
ls ./emissions/

# View FLOPS analysis
ls ./flops_logs/
```

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for datasets and checkpoints
- **Python**: 3.9+ with CUDA support

### Expected Training Time
- **COCO Training**: ~6-12 hours on RTX 4090
- **Robot Reasoning**: ~30 minutes additional per epoch
- **Full Pipeline**: ~8-15 hours depending on epochs and dataset size

## 🌟 Key Features

- **GRPO Reasoning**: Tiny-R1 style chain-of-thought reasoning with policy optimization
- **Robot Selection Intelligence**: Multi-step reasoning for selecting appropriate robots (Drone, Underwater Robot, Humanoid, Robot with Wheels, Robot with Legs)
- **Cross-Modal Fusion**: FIBER-inspired architecture for vision-language understanding
- **BitNet Quantization**: 1.58-bit quantized components for efficient inference
- **Episodic Memory**: Cross-modal memory system for visual-text associations
- **Comprehensive Training**: COCO dataset training with unlimited data
- **Multiple Image Format Support**: Supports JPG, JPEG, PNG, WEBP formats
- **Comprehensive Logging**: Detailed WandB visualizations and metrics tracking
- **Hugging Face Integration**: Automatic model uploads after training
- **Carbon Tracking**: Environmental impact monitoring

## 🏗️ Architecture

```text
BitMar Model with GRPO Reasoning Pipeline:

┌────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                       │
├────────────────────────────────┬───────────────────────────────────────────────┤
│        Text Input              │            Vision Input                       │
│    "Navigate indoor mall"      │         [Mall Image Data]                    │
│      [Token IDs: 512]          │        [RGB: 224x224x3]                      │
└────────────────────────────────┴───────────────────────────────────────────────┘
                 ↓                                    ↓
┌────────────────────────────────┐    ┌───────────────────────────────────────────┐
│      BITNET TEXT ENCODER       │    │        DINOV2 VISION ENCODER             │
│                                │    │                                           │
│  • 4 Transformer Layers        │    │  • Pretrained DiNOv2 Base               │
│  • 1.58-bit Quantization      │    │  • 768-dim Feature Output               │
│  • Multi-Head Attention       │    │  • Patch-based Processing               │
│  • Output: [B, 512, 768]      │    │  • Output: [B, 768]                     │
└────────────────────────────────┘    └───────────────────────────────────────────┘
                 ↓                                    ↓
┌────────────────────────────────────────────────────────────────────────────────┐
│                        FIBER CROSS-MODAL FUSION                               │
│                                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │  Vision Encoder  │  │  Text Encoder    │  │     Fusion Layers           │ │
│  │   [B, 768]       │  │  [B, 512, 768]   │  │  • 2 Transformer Layers     │ │
│  │      ↓           │  │       ↓          │  │  • 4 Attention Heads        │ │
│  │  Vision Proj     │  │   Text Proj      │  │  • Cross-Modal Attention    │ │
│  │   [B, 768]       │  │  [B, 512, 768]   │  │  • Output: [B, 512, 768]    │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
                                       ↓
╔════════════════════════════════════════════════════════════════════════════════╗
║                          GRPO REASONING MODULE                                ║
║                                                                                ║
║  ┌────────────────────────────────────────────────────────────────────────┐   ║
║  │                     CHAIN-OF-THOUGHT GENERATION                        │   ║
║  │                                                                        │   ║
║  │  Step 1: [Context Analysis]     → "Indoor navigation task"            │   ║
║  │  Step 2: [Environment Analysis] → "Crowded space, obstacles"          │   ║
║  │  Step 3: [Robot Evaluation]     → "Need human-like navigation"        │   ║
║  │  Step 4: [Selection Logic]      → "Humanoid best suited"              │   ║
║  │  Step 5: [Confidence Check]     → "High confidence: 0.95"             │   ║
║  │                                                                        │   ║
║  │  LSTM Continuity: [512-dim hidden state across steps]                 │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                       ↓                                        ║
║  ┌────────────────────────────────────────────────────────────────────────┐   ║
║  │                      ROBOT SELECTION POLICY                           │   ║
║  │                                                                        │   ║
║  │  Policy Network: [768] → [256] → [128] → [6 robots]                   │   ║
║  │  Value Network:  [768] → [256] → [128] → [1 value]                    │   ║
║  │                                                                        │   ║
║  │  Robot Probabilities:                                                  │   ║
║  │  • Drone: 0.05          • Underwater: 0.02                           │   ║
║  │  • Humanoid: 0.85       • Wheels: 0.03                               │   ║
║  │  • Legs: 0.04           • No Robot: 0.01                             │   ║
║  │                                                                        │   ║
║  │  Selected: HUMANOID (confidence: 0.85)                                │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
║                                       ↓                                        ║
║  ┌────────────────────────────────────────────────────────────────────────┐   ║
║  │                    GRPO POLICY OPTIMIZATION                           │   ║
║  │                                                                        │   ║
║  │  • Reward Calculation: R = task_success + reasoning_quality           │   ║
║  │  • Advantage Estimation: A = R - V(state)                             │   ║
║  │  • Policy Loss: -log π(robot|state) * A                               │   ║
║  │  • Value Loss: (R - V(state))²                                        │   ║
║  │  • Entropy Loss: -H(π) for exploration                                │   ║
║  │                                                                        │   ║
║  │  Output: Enhanced features [B, 512, 768] + Reasoning state [B, 768]   │   ║
║  └────────────────────────────────────────────────────────────────────────┘   ║
╚════════════════════════════════════════════════════════════════════════════════╝
                                       ↓
┌────────────────────────────────────────────────────────────────────────────────┐
│                         EPISODIC MEMORY SYSTEM                                │
│                                                                                │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐ │
│  │   Episode Store  │  │   Memory Query   │  │      Memory Retrieval       │ │
│  │                  │  │                  │  │                              │ │
│  │ • 32 Slots       │  │ • Query Creation │  │ • Attention-based Lookup    │ │
│  │ • 128-dim Each   │  │ • Text + Vision  │  │ • Contextual Enhancement     │ │
│  │ • LRU Updates    │  │ • Reasoning      │  │ • Memory Integration         │ │
│  │ • Cross-Modal    │  │   Enhanced       │  │ • Output: [B, 128]          │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌────────────────────────────────────────────────────────────────────────────────┐
│                        BITNET TEXT DECODER                                    │
│                                                                                │
│  • 4 Transformer Layers with 1.58-bit Quantization                           │
│  • Enhanced with Reasoning Context + Memory Context                           │
│  • Input: [B, 512, 768] (Reasoning + Memory Enhanced)                         │
│  • Multi-Head Attention with Reasoning Guidance                               │
│  • Output: Text Generation + Robot Selection Tokens                           │
└────────────────────────────────────────────────────────────────────────────────┘
                                       ↓
┌────────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                     │
│                                                                                │
│  Generated Text: "For navigating the indoor mall, I recommend using a        │
│                   humanoid robot due to its ability to navigate crowded       │
│                   spaces and interact naturally with people."                 │
│                                                                                │
│  Robot Selection: HUMANOID (Confidence: 0.85)                                │
│  Reasoning Quality: 0.92                                                      │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 🧠 GRPO-Based Chain-of-Thought Reasoning

BitMar implements a novel **Generalized Reinforcement Learning from Policy Optimization (GRPO)** reasoning system that combines chain-of-thought generation with policy gradient optimization for robot selection tasks. This implementation bridges symbolic reasoning with neural policy learning, enabling the model to perform structured multi-step reasoning while optimizing for task-specific outcomes.

### Theoretical Foundation

The GRPO reasoning module operates on the principle of **sequential decision-making under uncertainty** with **policy gradient optimization**. Unlike traditional chain-of-thought approaches that rely purely on next-token prediction, our system treats reasoning as a **Markov Decision Process (MDP)** where each reasoning step represents a state transition guided by learned policies.

**Core Architecture Components:**

- **Sequential Reasoning States**: The system maintains a sequence of hidden states `h_t ∈ ℝᵈ` across `T` reasoning steps, where each state encodes accumulated reasoning knowledge
- **Policy Networks**: Separate policy heads for reasoning token generation and robot selection actions, enabling dual optimization objectives
- **Value Function**: Estimates the expected future reward for each reasoning state, providing baseline for advantage computation
- **Attention-Based Robot Reasoning**: Multi-head attention mechanism that grounds robot selection in visual and textual context

### Mathematical Formulation

The GRPO reasoning process can be formalized through two key equations (omitted for GitHub compatibility):

**Chain-of-Thought State Evolution:**
*Formula omitted for compatibility.*

**GRPO Policy Optimization:**
*Formula omitted for compatibility.*

### Implementation Architecture

**Multi-Step Reasoning Pipeline:**

1. **Context Encoding**: Vision-language features from FIBER fusion are encoded into initial reasoning state
2. **LSTM-Based Continuity**: Bidirectional LSTM maintains reasoning coherence across steps with hidden state persistence
3. **Thought Generation**: BitNet-quantized generators produce intermediate reasoning thoughts with quality scoring
4. **Robot Attention**: Multi-head attention mechanism evaluates robot capabilities against current reasoning context
5. **Policy Optimization**: Separate policy and value networks optimize for both reasoning quality and robot selection accuracy

**Quality Assessment Framework:**

The system evaluates reasoning quality across four dimensions:
- **Coherence**: Logical consistency between reasoning steps
- **Relevance**: Alignment with task requirements and visual context  
- **Completeness**: Coverage of important decision factors
- **Accuracy**: Correctness of final robot selection

**Reward Signal Integration:**

- **Task Success Rewards**: Direct feedback from robot selection accuracy
- **Reasoning Quality Rewards**: Multi-dimensional assessment of thought quality
- **Advantage Estimation**: Temporal difference learning for value function updates
- **Entropy Regularization**: Encourages exploration during policy learning

### Scientific Contributions

**Novel Aspects of Our GRPO Implementation:**

- **Hybrid Symbolic-Neural Reasoning**: Combines structured reasoning steps with neural policy optimization
- **Multi-Modal Grounding**: Integrates vision, text, and robot capability knowledge in reasoning chain
- **Adaptive Step Weighting**: Dynamic importance weighting of reasoning steps based on learned value functions
- **Quality-Aware Training**: Explicit modeling and optimization of reasoning quality dimensions
- **Quantized Reasoning Components**: BitNet quantization applied to reasoning networks for efficiency

**Advantages Over Standard Chain-of-Thought:**

- **Goal-Oriented Optimization**: Direct optimization for task outcomes rather than just linguistic fluency
- **Uncertainty Quantification**: Value functions provide confidence estimates for reasoning decisions
- **Multi-Objective Learning**: Simultaneous optimization of reasoning quality and task performance
- **Adaptive Exploration**: Entropy regularization enables discovery of novel reasoning strategies
- **Cross-Modal Integration**: Seamless integration of visual and textual reasoning context

The GRPO reasoning system represents a significant advancement in neural reasoning architectures, providing both interpretable reasoning chains and optimized task performance through principled reinforcement learning integration.

## 🤖 Robot Selection Capabilities

The GRPO reasoning module can intelligently select robots based on task requirements:

- **Drone**: Aerial tasks, surveillance, fast movement, lightweight transport
- **Underwater Robot**: Marine exploration, underwater inspection, aquatic environments
- **Humanoid**: Complex manipulation, human interaction, tool use, indoor navigation
- **Robot with Wheels**: Fast ground movement, payload transport, warehouse operations
- **Robot with Legs**: Rough terrain navigation, stairs, outdoor exploration, search & rescue

### GRPO Reasoning Template

Here's how the GRPO module processes a robot selection task using structured reasoning tags:

```xml
📋 GRPO REASONING TEMPLATE

Input: "Survey a construction site and identify safety hazards"
Vision: [Construction site image with cranes, workers, equipment]

<thinking>
This is a surveillance and inspection task that requires getting an overview of a large construction area. Construction sites typically have multiple levels, moving equipment, and various safety hazards that need to be identified from a safe distance. The robot needs to cover a large area efficiently while maintaining safety protocols.
</thinking>

<environment_analysis>
Construction sites have elevated structures, moving heavy equipment, limited ground access, safety restrictions for human operators, and potential hazards like falling objects. The terrain may be uneven with obstacles and restricted access zones.
</environment_analysis>

<robot_evaluation>
- Drone: Can provide aerial perspective, obstacle avoidance, equipped with cameras, fast coverage, minimal disruption
- Humanoid: Limited by safety restrictions, slower coverage, risk from falling objects
- Underwater Robot: Not applicable for this environment
- Robot with Wheels: Limited by terrain, cannot access elevated areas
- Robot with Legs: Better than wheels but still ground-limited, safety risks
- No Robot: Manual inspection would be dangerous and inefficient
</robot_evaluation>

<selection_reasoning>
For construction site survey and safety hazard identification, a drone is optimal because:
1. Aerial perspective allows comprehensive site overview
2. Safe distance from ground-level hazards
3. Can access all levels and hard-to-reach areas
4. Equipped with high-resolution cameras and sensors
5. Fast coverage with minimal disruption to ongoing work
6. Can hover and maintain stable position for detailed inspection
</selection_reasoning>

<confidence_check>
High confidence in drone selection (0.89). Alternative would be humanoid robot with extensive safety equipment, but this introduces unnecessary risk and significantly slower coverage. Drone clearly provides the best safety-to-efficiency ratio for this task.
</confidence_check>

<robot_selection>
Selected Robot: Drone
Confidence: 0.89
Justification: Optimal for aerial surveillance with comprehensive coverage and minimal safety risk
</robot_selection>
```

**More Examples:**

```xml
Input: "Delicate laboratory equipment manipulation"

<thinking>
This task requires precise manipulation of sensitive laboratory equipment. The environment is controlled, indoors, and requires human-like dexterity and careful handling to avoid damage to expensive instruments.
</thinking>

<robot_evaluation>
- Humanoid: Excellent dexterity, can use existing tools, precise manipulation
- Robot with Wheels: Limited manipulation capabilities
- Robot with Legs: Better mobility but less manipulation focus
- Drone: Cannot perform delicate manipulation tasks
</robot_evaluation>

<robot_selection>
Selected Robot: Humanoid
Confidence: 0.92
Justification: Superior manipulation capabilities essential for delicate laboratory work
</robot_selection>
```

```xml
Input: "Inspect underwater pipeline for damage"

<thinking>
This is clearly an aquatic environment task that requires underwater navigation and inspection capabilities. No other robot type can operate in this environment effectively.
</thinking>

<robot_selection>
Selected Robot: Underwater Robot
Confidence: 0.98
Justification: Only robot capable of underwater operation and pipeline inspection
</robot_selection>
```

### Example Robot Reasoning

- **"Inspect underwater pipelines"** → **Underwater Robot**
- **"Survey construction site from above"** → **Drone**
- **"Navigate crowded indoor mall"** → **Humanoid**
- **"Transport heavy equipment across terrain"** → **Robot with Legs**
- **"Delicate laboratory manipulation"** → **Humanoid**

## 🛠️ Installation

```bash
git clone https://github.com/euhidaman/Tiny-BitGen.git
cd Tiny-BitGen
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📦 Download COCO Dataset

First, download the COCO dataset with captions:

```bash
# Download COCO dataset from Kaggle
python download_coco_supplement.py
```

This will:
- Download COCO images and captions from Kaggle
- Process and align image-caption pairs
- Support multiple image formats (JPG, PNG, WEBP, etc.)
- Prepare data for unlimited training

## 🚀 Training Commands

### Standard COCO Training with GRPO Reasoning

```bash
# Train on full COCO dataset with GRPO reasoning enabled
python train.py --config configs/bitmar_coco.yaml

# Training with specific GPU device
python train.py --config configs/bitmar_coco.yaml --device cuda:0
```

### Training with Custom Options

```bash
# Save checkpoint every 1000 steps
python train.py --config configs/bitmar_coco.yaml --save_every_n_steps 1000

# Enable/disable evaluation phases
python train.py --config configs/bitmar_coco.yaml --enable_fast_eval --disable_full_eval

# Complete training with all options
python train.py \
    --config configs/bitmar_coco.yaml \
    --device cuda:0 \
    --save_every_n_steps 1000 \
    --enable_fast_eval \
    --enable_full_eval
```

## 🧠 GRPO Reasoning Configuration

Configure GRPO reasoning in `configs/bitmar_coco.yaml`:

```yaml
grpo_reasoning:
  enabled: true                    # Enable GRPO reasoning
  max_reasoning_steps: 5           # Chain-of-thought steps
  reasoning_temperature: 0.7       # Generation temperature
  reasoning_weight: 0.3            # Feature blending weight
  loss_weight: 0.2                 # Loss contribution weight
  training:
    learning_rate: 1e-6            # GRPO learning rate
    value_loss_coef: 0.5           # Value function loss weight
    entropy_coef: 0.01             # Entropy regularization
    max_grad_norm: 0.5             # Gradient clipping
    reward_scaling: 1.0            # Reward scaling factor
```

## 🧪 Testing GRPO Integration

The GRPO reasoning system is fully integrated into the main training pipeline. All reasoning capabilities are tested during training.

## 📊 Model Usage Examples

### Extract Robot Selections

```python
import torch
from src.model import create_bitmar_model

# Load model
config = {...}  # Your config
model = create_bitmar_model(config)

# Forward pass
outputs = model(input_ids, attention_mask, vision_features)

# Extract robot selections
robot_selections = model.extract_robot_selections(outputs)
print(robot_selections)  # ["Drone", "Underwater Robot, Robot with Legs"]
```

### Get Reasoning Analysis

```python
# Get detailed reasoning analysis
analysis = model.get_reasoning_analysis(outputs)

print(f"Robot probabilities: {analysis['robot_probabilities']}")
print(f"Reasoning quality: {analysis['reasoning_quality']}")
print(f"Robot selections: {analysis['robot_selections']}")
```

## 📊 WandB Logging & Visualizations

BitMar includes comprehensive logging to Weights & Biases with detailed visualizations and metrics tracking.

### 🎯 Training Metrics

**train/loss**
- **What**: Cross-entropy loss during training
- **X-axis**: Training steps
- **Y-axis**: Loss value
- **Interpretation**: Should decrease over time; sudden spikes indicate potential issues

**train/learning_rate**
- **What**: Learning rate schedule (cosine annealing with warm restarts)
- **X-axis**: Training steps
- **Y-axis**: Learning rate value
- **Interpretation**: Shows learning rate cycles; restarts help escape local minima

**train/cross_modal_similarity**
- **What**: Cosine similarity between text and vision features
- **X-axis**: Training steps
- **Y-axis**: Similarity score (-1 to 1)
- **Interpretation**: Higher values = better cross-modal alignment; key metric for multimodal understanding

**train/grpo_reasoning_loss** (NEW)
- **What**: GRPO reasoning loss for robot selection
- **X-axis**: Training steps
- **Y-axis**: Loss value
- **Interpretation**: Should decrease as robot selection reasoning improves

### 🤖 GRPO Reasoning Metrics

**reasoning/robot_selection_accuracy**
- **What**: Accuracy of robot selection predictions
- **X-axis**: Training steps
- **Y-axis**: Accuracy percentage
- **Interpretation**: Should increase as reasoning quality improves

**reasoning/thought_scores**
- **What**: Quality scores of reasoning steps
- **X-axis**: Training steps
- **Y-axis**: Score distribution
- **Interpretation**: Higher scores indicate better reasoning quality

**reasoning/robot_probabilities**
- **What**: Distribution of robot selection probabilities
- **X-axis**: Training steps
- **Y-axis**: Probability values for each robot type
- **Interpretation**: Shows which robots are selected most frequently

### 📈 Epoch-Level Metrics

**epoch/train_loss**
- **What**: Average loss per epoch
- **X-axis**: Epoch number
- **Y-axis**: Loss value
- **Interpretation**: Should show steady decrease across epochs

**epoch/cross_modal_similarity**
- **What**: Average cross-modal similarity per epoch
- **X-axis**: Epoch number
- **Y-axis**: Similarity score
- **Interpretation**: Should increase as model learns better alignment

**epoch/grpo_reasoning_quality** (NEW)
- **What**: Average reasoning quality per epoch
- **X-axis**: Epoch number
- **Y-axis**: Quality score
- **Interpretation**: Should increase as reasoning improves

### 🧠 Memory & Attention Analysis

**Memory/Usage_Mean, Memory/Usage_Max, Memory/Usage_Min**: Episodic memory slot utilization statistics
**Memory/Active_Slots_Percentage**: Percentage of memory slots being actively used  
**Memory/Analysis_Avg_Similarity**: Average similarity between active memory slots (lower = more diverse)

**Attention/CrossModal_layer_X_Mean, Attention/CrossModal_layer_X_Max**: Cross-modal attention weights by layer
**Attention/Memory_Mean, Attention/Memory_Max**: Memory attention patterns

### ⚡ Quantization Metrics

**Quantization/BitNet_Layer_Count**: Number of quantized layers in the model
**Quantization/Sparsity_Mean**: Average sparsity percentage across all BitNet layers (30-60% typical)
**Quantization/Distribution_Balance**: Balance between +1/-1 weights (closer to 1.0 = better balance)
**Quantization/Compression_Effectiveness**: Overall compression achieved through sparsity

### 🤗 Hugging Face Integration

**huggingface/upload_success**: Whether model upload to HF Hub succeeded
**huggingface/repo_url**: Link to uploaded model repository

## 📂 Project Structure

```
Tiny-BitGen/
├── src/
│   ├── model.py                        # Main BitMar model with GRPO integration
│   ├── grpo_reasoning_module.py         # GRPO reasoning implementation
│   ├── fiber_fusion.py                 # Cross-modal fusion
│   ├── coco_dataset.py                 # COCO dataset handling
│   ├── wandb_logger.py                 # WandB integration
│   └── ...
├── configs/
│   └── bitmar_coco.yaml               # Main training configuration
├── train.py                           # Main training script
├── download_coco_supplement.py        # COCO dataset download
├── README.md                          # Project documentation
└── requirements.txt                   # Dependencies
```

## 🎯 Performance Targets

- **Model Size**: ~50MB (BitNet quantization)
- **Cross-Modal Similarity**: >0.75
- **Robot Selection Accuracy**: >80%
- **Reasoning Quality**: >0.8
- **Memory Efficiency**: >0.8

## 🔧 Configuration Files

### Main Config: `configs/bitmar_coco.yaml`
- Model architecture settings
- GRPO reasoning configuration
- Training hyperparameters
- Data processing options
- Logging and evaluation settings

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download COCO dataset**:
   ```bash
   python download_coco_supplement.py
   ```

3. **Start training**:
   ```bash
   python train.py --config configs/bitmar_coco.yaml
   ```

The GRPO reasoning is automatically integrated and trained with the main model.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔒 Security & Robustness Features

BitMar implements comprehensive security measures to ensure stable and safe operation in production environments:

### 🛡️ Numerical Stability Protection

**Quantization Security**:
- **Scale Clamping**: Weight and activation scales are bounded between `1e-5` and `1e3` to prevent extreme scaling that could lead to numerical instability or overflow attacks
- **Gradient Clipping**: Weight normalization is clipped to `±10.0` range to prevent exploding gradients and potential DoS through resource exhaustion
- **Division by Zero Protection**: All quantization operations include safeguards against division by zero with minimum thresholds (`1e-8`)
- **Activation Bounds**: Input activations are clamped to `±1e6` range to prevent overflow and numerical instability

**BitNet 1.58-bit Quantization Hardening**:
- Ternary quantization uses fixed thresholds (`2/3`) to prevent manipulation through adversarial weight perturbations
- Straight-through estimator maintains gradient flow security during training
- Quantization parameters are registered as buffers to prevent unauthorized modification

### 🔍 Input Validation & Dimension Safety

**Strict Dimension Validation**:
- All attention mechanisms validate query, key, and value tensor dimensions before processing
- Cross-modal fusion validates vision and text feature compatibility
- Memory system validates episode dimensions against expected schemas
- GRPO reasoning validates input feature dimensions for robot selection

**Runtime Checks**:
- Memory loading includes metadata validation to prevent corrupted or malicious memory injection
- Vision feature dimensions are validated against model expectations
- Attention mask shapes are verified and properly broadcast to prevent tensor misalignment

### 🧠 Memory System Security

**Episodic Memory Protection**:
- Memory slot access is bounded by predefined limits (32 slots by default)
- LRU (Least Recently Used) eviction prevents memory overflow attacks
- Cross-modal memory queries include dimension compatibility checks
- Memory persistence includes metadata verification to prevent tampering

**Memory Isolation**:
- Each episode is stored with integrity checks
- Memory retrieval includes attention-based validation
- External memory loading includes compatibility verification before integration

### 🤖 GRPO Reasoning Security

**Robot Selection Validation**:
- Selection probabilities are normalized and clamped to valid ranges (`0-1`)
- Confidence thresholds prevent low-quality robot selections (minimum `0.3`)
- Chain-of-thought generation includes quality scoring to detect malformed reasoning
- Policy networks include value function validation to prevent reward hacking

**Reasoning Quality Assurance**:
- Multi-step reasoning includes coherence, relevance, completeness, and accuracy validation
- Temperature controls prevent both overly deterministic and overly random selections
- LSTM continuity maintains reasoning state integrity across steps

### ⚡ Production Safety Features

**Resource Management**:
- GPU memory is managed with automatic cleanup and error handling
- Training includes gradient norm monitoring to detect instability
- Model checkpoint validation ensures integrity before loading
- Background processes include proper exception handling

**Error Recovery**:
- Graceful degradation when optional components (GRPO) are unavailable
- Comprehensive logging for security event monitoring
- Automatic fallback mechanisms for failed operations
- Input sanitization prevents malformed data from causing crashes

**Model Integrity**:
- Hugging Face integration includes upload verification
- Carbon tracking prevents excessive resource consumption
- WandB logging includes security metrics monitoring
- Configuration validation ensures only safe hyperparameters are used

These security measures ensure BitMar operates safely in production environments while maintaining high performance and preventing common attack vectors like adversarial inputs, resource exhaustion, and model manipulation.

## 🙏 Acknowledgments

- BitNet quantization for efficient neural networks
- DiNOv2 for robust vision features
- FIBER for cross-modal fusion inspiration
- Tiny-R1 for chain-of-thought reasoning patterns
- GRPO for policy optimization in reasoning tasks
