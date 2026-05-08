"""CAFE-specific networks and utils.

CAFE (Wang et al., CVPR 2022) requires a forward pass that returns
``(logits, [layer_features])`` for multi-layer feature alignment, so it
cannot share the standard feature-less ConvNet used by DC/DM/IDC.

This sub-package therefore ships its own ``networks`` and ``utils`` modules
(``get_network``, ``epoch``, ``evaluate_synset`` etc. that match the CAFE
forward contract). Shared dataset utilities still live in
``data_handler/`` at the project root.
"""
