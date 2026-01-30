//! EmbeddingNetwork implementation

use super::types::*;
use crate::{Error, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

impl EmbeddingNetwork {
    /// Forward pass through the network
    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        // Apply convolutional layers if present
        for conv_layer in &self.conv_layers {
            x = conv_layer.forward(&x)?;
            x = x.relu()?;
        }

        // Flatten for fully connected layers
        if !self.conv_layers.is_empty() {
            let shape = x.shape();
            let batch_size = shape.dims()[0];
            let flattened_size = shape.elem_count() / batch_size;
            x = x.reshape(&[batch_size, flattened_size])?;
        }

        // Apply fully connected layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // Apply ReLU activation except for the last layer
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }

        Ok(x)
    }
}
