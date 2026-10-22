"""Ordinary explicit-axis persistent edit; not a learned editor or full recorder repair."""
import jax.numpy as jnp
def persistent_context_edit(context, dimension, delta):
    if context.ndim != 3:raise ValueError('Expected batch,time,context-dimension tensor')
    if not isinstance(dimension,int) or not 0<=dimension<context.shape[-1]:raise ValueError('Invalid context dimension')
    shift=jnp.asarray(delta,dtype=context.dtype)
    if shift.ndim==0:shift=jnp.broadcast_to(shift,(context.shape[0],))
    if shift.shape!=(context.shape[0],):raise ValueError('Delta must be scalar or one fixed value per batch member')
    return context.at[:,:,dimension].add(shift[:,None])
