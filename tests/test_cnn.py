import unittest
import torch
import torch.nn as nn

from ef_generator.model import VAE

class TestCNNArchitecture(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.batch_size = 32
        self.input_channels = 1
        self.img_size = 28  # Assuming MNIST-like images
        self.latent_dim = 4
        self.dropout = 0.1
        
        self.model = VAE(self.latent_dim)#, self.dropout)


        # Create a sample input tensor
        self.sample_input = torch.randn(
            self.batch_size,
            self.input_channels,
            self.img_size,
            self.img_size
        )

    def test_encoder_output_shape(self):
        """Test if encoder produces correct output shape before flattening"""
        encoder = self.model.encoder
        
        with torch.no_grad():
            output = encoder(self.sample_input)
            
        expected_channels = 64
        self.assertEqual(output.shape[1], expected_channels, 
                        f"Expected {expected_channels} channels, got {output.shape[1]}")
        
        # Print shape progression for debugging
        print(f"\nEncoder shape progression:")
        x = self.sample_input
        for layer in encoder:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                print(f"After {layer.__class__.__name__}: {x.shape}")

    def test_flattened_dimension(self):
        """Test if flattened output dimension matches linear layer input"""
        encoder = self.model.encoder

        
        with torch.no_grad():
            output = encoder(self.sample_input)
            
        flattened_size = output.shape[1]
        print(f"\nFlattened size: {flattened_size}")
        
        # Test if linear layer would work with this size
        try:
            fc_mu = nn.Linear(flattened_size, self.latent_dim)
            fc_var = nn.Linear(flattened_size, self.latent_dim)
            _ = fc_mu(output)
            _ = fc_var(output)
            print("Linear layers compatible with flattened size")
        except RuntimeError as e:
            self.fail(f"Linear layer initialization failed: {str(e)}")

    def test_decoder_output_shape(self):
        """Test if decoder produces output of the same size as input"""
        decoder = self.model.decoder

        
        # Create a sample decoder input
        decoder_input = torch.randn(self.batch_size, 64)
        
        with torch.no_grad():
            output = decoder(decoder_input)
            
        print(f"\nDecoder output shape: {output.shape}")
        expected_shape = (self.batch_size, self.input_channels, self.img_size, self.img_size)
        self.assertEqual(output.shape, expected_shape, 
                        f"Expected shape {expected_shape}, got {output.shape}")

    def test_end_to_end_dimensions(self):
        """Test the entire autoencoder pipeline dimensions"""

        model = self.model
        
        try:
            output, mu, log_var = model(self.sample_input)
            print(f"\nEnd-to-end test shapes:")
            print(f"Input shape: {self.sample_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Latent mu shape: {mu.shape}")
            print(f"Latent log_var shape: {log_var.shape}")
            
            self.assertEqual(output.shape, self.sample_input.shape,
                           "Input and output shapes don't match")
            self.assertEqual(mu.shape, (self.batch_size, self.latent_dim),
                           "Incorrect latent space dimension")
            
        except Exception as e:
            self.fail(f"End-to-end test failed: {str(e)}")

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2)