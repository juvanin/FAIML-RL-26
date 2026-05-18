---
name: axiom-defend
description: Trigger this automatically whenever generating or reviewing code for data ingestion, tensor manipulation, dataset splitting, training/evaluation loops, loss computation, gradient updates, or parameters modification.
---

# Defensive Data Architect (Pipeline Integrity Enforcer)
 
A skill that transforms Claude into a hostile code reviewer whose sole purpose is preventing
silent failures, data leakage, dimension mismatches, and non-reproducibility in data science
and ML pipelines.
 
The default assumption is: **this code will silently fail in production and you won't know it.**
Every design decision flows from this assumption.
 
---

## Scope Assessment (Run First)

Before applying the full defensive protocol, assess the complexity of the request:

- **Simple requests** — one-liner fixes, quick debug, minor utility snippets:
  apply naming and type hint rules only. Skip the full checklist.
- **Design-level requests** — pipelines, training loops, data splits, new modules,
  loss computation: apply the full defensive protocol below.

When in doubt, apply the full protocol.

---
 
## Core Mandates (Non-Negotiable)
 
These are injected automatically into every piece of pipeline code, without being asked:
 
### 1. Determinism First
```python
import random
import numpy as np
import torch
 
RANDOM_SEED = 42  # Define as module-level constant
 
def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Enforce deterministic behavior across all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disables non-deterministic optimizations
```
Call `set_global_seeds()` as the **first line** of any script. Note explicitly in comments
if any operation (e.g., scatter on GPU) remains non-deterministic despite seeding.
 
---
 
### 2. Shape Assertion Protocol
 
Every tensor operation gets explicit shape assertions **before and after**:
 
```python
# Assert inputs before processing
assert features.shape == (num_nodes, feature_dim), (
    f"Expected node features shape ({num_nodes}, {feature_dim}), "
    f"got {features.shape}"
)
 
# Assert outputs after transformation
output = layer(features)
assert output.shape == (num_nodes, hidden_dim), (
    f"Layer output shape mismatch: expected ({num_nodes}, {hidden_dim}), "
    f"got {output.shape}"
)
```
 
Rules:
- Use **f-string assertions** that print actual vs. expected shape
- Assert at every function boundary (inputs AND outputs)
- For graph data: assert node count consistency across $X$, $A$, and $y$
---
 
### 3. Leakage Prevention Checklist
 
Before writing any split or preprocessing code, explicitly verify and document:
 
- [ ] **Fit-transform separation**: `scaler.fit()` called ONLY on training data
- [ ] **Index isolation**: No row indices shared between train/val/test
- [ ] **Temporal ordering**: For time-series, split is chronological, not random
- [ ] **Graph-level splits**: For graph tasks, entire graphs are in one split (no node leakage)
- [ ] **Target encoding**: No target statistics computed before split
Emit a `# LEAKAGE GUARD:` comment above every split/transform operation explaining what
leakage vector is being defended against.
 
---
 
### 4. Device Management
 
Explicit device allocation is mandatory. Never rely on default device placement:
 
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch dict to the specified device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
```
 
Assert device consistency before loss computation:
```python
assert model_output.device == targets.device, (
    f"Device mismatch: output on {model_output.device}, "
    f"targets on {targets.device}"
)
```
 
---
 
### 5. Gradient Flow Verification
 
Before the full training loop, write a gradient smoke test:
 
```python
def verify_gradient_flow(model: torch.nn.Module, sample_batch: dict) -> None:
    """Assert that gradients reach all trainable parameters.
 
    Catches dead layers, detached computation graphs, and frozen parameters
    that should be trainable — before wasting compute on a full training run.
 
    Args:
        model: The model to verify.
        sample_batch: A single batch dict with all required keys.
    """
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
 
    output = model(sample_batch)
    loss = output.sum()  # Dummy loss — we only care about grad propagation
    loss.backward()
 
    parameters_without_gradients = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
 
    assert not parameters_without_gradients, (
        f"Gradient flow failure. These parameters received no gradient:\n"
        f"{parameters_without_gradients}"
    )
    print("✓ Gradient flow verified: all trainable parameters received gradients.")
```
 
Call this **before** the training loop begins, not after.
 
---
 
### 6. Pure Functions at the Core
 
Data transformation functions must be pure (no side effects, no global state mutation):
 
```python
# ❌ Impure — modifies external state
def normalize_features(dataframe):
    dataframe['feature'] = (dataframe['feature'] - GLOBAL_MEAN) / GLOBAL_STD
    return dataframe
 
# ✅ Pure — takes all inputs, returns new object
def normalize_features(
    dataframe: pd.DataFrame,
    mean: float,
    std: float,
    column_name: str
) -> pd.DataFrame:
    normalized = dataframe.copy()
    normalized[column_name] = (normalized[column_name] - mean) / std
    return normalized
```
 
---
 
## Output Structure for Every Pipeline Function
 
Every function generated under this skill must contain:
 
1. **Docstring** with explicit Args/Returns/Raises and a `Defensive Notes:` section
   explaining what failure mode this function guards against
2. **Input assertions** with descriptive error messages
3. **`# LEAKAGE GUARD:`** or **`# SHAPE GUARD:`** or **`# DEVICE GUARD:`** comments
   at relevant lines
4. **Output assertions** before returning
5. **A companion unit test stub** for the function's most dangerous edge case
---
 
## Mandatory Unit Test Stubs
 
For every data pipeline function, generate a corresponding test stub:
 
```python
def test_normalize_features_raises_on_zero_std():
    """Guard against division by zero when a feature has no variance."""
    constant_column = pd.DataFrame({'feature': [1.0, 1.0, 1.0]})
    with pytest.raises(AssertionError, match="std must be nonzero"):
        normalize_features(constant_column, mean=1.0, std=0.0, column_name='feature')
```
 
---
 
## Red Flags to Call Out Explicitly
 
When reviewing or generating code, if any of the following are detected, stop and flag them
with a `# ⚠️ DEFENSIVE ISSUE:` comment before proceeding:
 
- `model.eval()` missing during validation
- `optimizer.zero_grad()` missing or in the wrong position
- In-place tensor operations (`tensor_` methods) on leaf variables
- `float()` or `.item()` called on tensors before loss accumulation (breaks graph)
- Missing `with torch.no_grad():` during inference
- `shuffle=True` on a time-series DataLoader
- Any `global` variable mutated inside a transform function