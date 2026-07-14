==================
Training utilities
==================

.. automodule:: gromo.utils.training_utils
   :no-index:

.. currentmodule:: gromo.utils.training_utils

--------------
Growth updates
--------------

``compute_fixed_point_updates`` combines statistics accumulation and optimal-update
computation. Passing a :class:`FixedPointConfig` makes it recompute the complete
statistics dataloader at every iteration while solving the joint Gromo equation

.. math::

   D = \operatorname{OptimalDelta}(W - D).

The minus sign matches Gromo's convention: ``optimal_delta_layer`` is subtracted
when a proposal is previewed or applied. The final proposal remains stored on the
model, while the original parameters are restored.

.. autosummary::
    :toctree: generated/
    :template: function.rst

    compute_fixed_point_updates

.. currentmodule:: gromo.utils.fixed_point

.. autosummary::
    :toctree: generated/
    :template: class.rst

    FixedPointConfig
    FixedPointIteration
    FixedPointResult
    FixedPointUpdateResult

-------
Example
-------

.. code-block:: python

    from gromo.utils.fixed_point import FixedPointConfig
    from gromo.utils.training_utils import compute_fixed_point_updates

    result = compute_fixed_point_updates(
        model,
        train_loader,
        loss_function=loss_sum,
        fixed_point_config=FixedPointConfig(damping=0.5),
    )

Each iteration is a full statistics pass. For deterministic evaluations, use a
deterministic dataloader or supply ``dataloader_seed``. Persistent workers and
external random transforms may require additional control by the caller.
