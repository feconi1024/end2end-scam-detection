"""
Core modules for the differentiable cascaded end-to-end scam detector.

Submodules:
- dataset: TeleAntiFraudDataset and config loader
- projector: ModalityProjector bridging acoustic and semantic spaces
- model: DifferentiableCascadeModel (Whisper encoder + projector + BERT)
- loss: Joint CTC + classification loss
- trainer: Training loop with iterative freezing
"""

