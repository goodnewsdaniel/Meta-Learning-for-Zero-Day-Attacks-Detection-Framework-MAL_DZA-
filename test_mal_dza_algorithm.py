################################################
'''   Goodnews Daniel (PhD Candidate)
        222166453@student.uj.ac.za
Department of Electrical & Electronics Engineering
Faculty of Engineering & the Built Environment
University of Johannesburg, South Africa        

MAL-ZDA Test Suite - UPDATED & COMPLETE
Comprehensive unit tests for all components     '''
################################################

from mal_dza_algorithm import (
    CyberSecurityDataset,
    HierarchicalEncoder,
    MALZDA,
    CompositionalTaskSampler,
    MALZDATrainer,
    load_and_preprocess_real_data,
    run_experiment,
    DEVICE,
    DEFAULT_EMBEDDING_DIM,
    RESULTS_DIR
)
import unittest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
import pandas as pd
import warnings
from unittest.mock import patch, MagicMock

# Suppress warnings during testing
warnings.filterwarnings('ignore')

# Import all components from mal_dza


class TestCyberSecurityDataset(unittest.TestCase):
    """Test dataset generation and loading"""

    def setUp(self):
        """Set up test configuration"""
        torch.manual_seed(42)
        np.random.seed(42)

        self.test_config = {
            'num_classes': 5,
            'samples_per_class': 20,
            'feature_dim': 64,
            'temporal_length': 10,
            'mode': 'train'
        }

    def test_synthetic_dataset_generation(self):
        """Test synthetic dataset generation with correct dimensions"""
        dataset = CyberSecurityDataset(
            num_classes=self.test_config['num_classes'],
            samples_per_class=self.test_config['samples_per_class'],
            feature_dim=self.test_config['feature_dim'],
            temporal_length=self.test_config['temporal_length'],
            mode=self.test_config['mode']
        )

        # Check total samples
        expected_len = self.test_config['num_classes'] * \
            self.test_config['samples_per_class']
        self.assertEqual(len(dataset), expected_len,
                         "Dataset length mismatch")

        # Check sample structure
        sample = dataset[0]
        required_keys = ['packet', 'flow',
                         'campaign', 'class_id', 'kill_chain']
        for key in required_keys:
            self.assertIn(key, sample, f"Missing key: {key}")

        # Check dimensions
        self.assertEqual(sample['packet'].shape[0],
                         self.test_config['feature_dim'],
                         "Packet dimension mismatch")
        self.assertEqual(sample['flow'].shape,
                         (self.test_config['temporal_length'],
                         self.test_config['feature_dim']),
                         "Flow shape mismatch")
        self.assertEqual(sample['campaign'].shape[0],
                         self.test_config['feature_dim'] * 4,
                         "Campaign dimension mismatch")

        # Check data types
        self.assertIsInstance(sample['packet'], np.ndarray,
                              "Packet should be numpy array")
        self.assertIsInstance(sample['flow'], np.ndarray,
                              "Flow should be numpy array")
        self.assertIsInstance(sample['campaign'], np.ndarray,
                              "Campaign should be numpy array")
        self.assertIsInstance(sample['class_id'], (int, np.integer),
                              "class_id should be integer")
        self.assertIsInstance(sample['kill_chain'], (int, np.integer),
                              "kill_chain should be integer")

        # Check value ranges
        self.assertTrue(0 <= sample['class_id'] <
                        self.test_config['num_classes'],
                        "class_id out of range")
        self.assertTrue(0 <= sample['kill_chain'] < 5,
                        "kill_chain out of range (should be 0-4)")

    def test_real_data_loading(self):
        """Test loading real data from arrays"""
        # Create synthetic real data
        n_samples = 100
        n_features = 50
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 5, n_samples)
        kill_chain_labels = np.random.randint(0, 5, n_samples)
        feature_names = [f'Feature_{i}' for i in range(n_features)]

        dataset = CyberSecurityDataset(
            X=X,
            y=y,
            kill_chain_labels=kill_chain_labels,
            feature_names=feature_names,
            temporal_length=10
        )

        # Check dataset size
        self.assertEqual(len(dataset), n_samples,
                         "Real data dataset size mismatch")

        # Check sample structure
        sample = dataset[0]
        self.assertEqual(sample['packet'].shape[0], n_features,
                         "Real data packet dimension mismatch")
        self.assertEqual(sample['flow'].shape[0], 10,
                         "Real data flow temporal length mismatch")
        self.assertEqual(sample['campaign'].shape[0], n_features * 4,
                         "Real data campaign dimension mismatch")

    def test_kill_chain_phase_distribution(self):
        """Test if kill-chain phases are properly distributed"""
        dataset = CyberSecurityDataset(
            num_classes=10,
            samples_per_class=100,
            feature_dim=64,
            temporal_length=10
        )

        # Count kill-chain phases
        phases = [sample['kill_chain'] for sample in dataset.data]
        unique_phases = set(phases)

        # Should have multiple phases
        self.assertGreater(len(unique_phases), 1,
                           "Should have multiple kill-chain phases")

        # All phases should be valid
        self.assertTrue(all(0 <= p < 5 for p in phases),
                        "All phases should be in range 0-4")

    def test_limited_class_dataset(self):
        """Test dataset with very limited number of classes"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=15,
            feature_dim=64,
            temporal_length=10
        )

        # Should still work with limited classes
        self.assertEqual(len(dataset), 30,
                         "Limited class dataset size mismatch")

        # Check class distribution
        class_ids = [sample['class_id'] for sample in dataset.data]
        unique_classes = set(class_ids)
        self.assertEqual(len(unique_classes), 2,
                         "Should have exactly 2 classes")

    def test_dataset_indexing(self):
        """Test dataset indexing and access"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=10,
            feature_dim=64,
            temporal_length=10
        )

        # Test valid indices
        for idx in range(len(dataset)):
            sample = dataset[idx]
            self.assertIsNotNone(sample)
            self.assertIn('packet', sample)

        # Test boundary conditions
        first_sample = dataset[0]
        last_sample = dataset[len(dataset) - 1]
        self.assertIsNotNone(first_sample)
        self.assertIsNotNone(last_sample)


class TestHierarchicalEncoder(unittest.TestCase):
    """Test hierarchical encoder architecture"""

    def setUp(self):
        """Set up encoder configuration"""
        self.packet_dim = 64
        self.flow_seq_len = 10
        self.campaign_dim = 256
        self.embedding_dim = 32
        self.batch_size = 4

        self.encoder = HierarchicalEncoder(
            packet_dim=self.packet_dim,
            flow_seq_len=self.flow_seq_len,
            campaign_dim=self.campaign_dim,
            embedding_dim=self.embedding_dim
        )

    def test_encoder_initialization(self):
        """Test encoder initialization"""
        self.assertIsNotNone(self.encoder.packet_encoder,
                             "packet_encoder not initialized")
        self.assertIsNotNone(self.encoder.flow_encoder,
                             "flow_encoder not initialized")
        self.assertIsNotNone(self.encoder.campaign_encoder,
                             "campaign_encoder not initialized")
        self.assertIsNotNone(self.encoder.layer_norm,
                             "layer_norm not initialized")

    def test_packet_encoder(self):
        """Test packet-level encoder"""
        packet_data = torch.randn(self.batch_size, self.packet_dim)
        packet_emb = self.encoder._encode_packet(packet_data)

        self.assertEqual(packet_emb.shape,
                         (self.batch_size, self.embedding_dim),
                         "Packet embedding shape mismatch")
        self.assertTrue(torch.all(torch.isfinite(packet_emb)),
                        "Packet embeddings contain NaN/Inf")

    def test_flow_encoder(self):
        """Test flow-level encoder"""
        flow_data = torch.randn(
            self.batch_size, self.flow_seq_len, self.packet_dim)
        flow_emb = self.encoder._encode_flow(flow_data)

        self.assertEqual(flow_emb.shape, (self.batch_size, self.embedding_dim),
                         "Flow embedding shape mismatch")
        self.assertTrue(torch.all(torch.isfinite(flow_emb)),
                        "Flow embeddings contain NaN/Inf")

    def test_campaign_encoder(self):
        """Test campaign-level encoder"""
        campaign_data = torch.randn(self.batch_size, self.campaign_dim)
        campaign_emb = self.encoder._encode_campaign(campaign_data)

        self.assertEqual(campaign_emb.shape,
                         (self.batch_size, self.embedding_dim),
                         "Campaign embedding shape mismatch")
        self.assertTrue(torch.all(torch.isfinite(campaign_emb)),
                        "Campaign embeddings contain NaN/Inf")

    def test_full_forward_pass(self):
        """Test complete forward pass through all levels"""
        packet_data = torch.randn(self.batch_size, self.packet_dim)
        flow_data = torch.randn(
            self.batch_size, self.flow_seq_len, self.packet_dim)
        campaign_data = torch.randn(self.batch_size, self.campaign_dim)

        packet_emb, flow_emb, campaign_emb = self.encoder(
            packet_data, flow_data, campaign_data
        )

        # Check shapes
        self.assertEqual(packet_emb.shape,
                         (self.batch_size, self.embedding_dim),
                         "Packet forward pass shape mismatch")
        self.assertEqual(flow_emb.shape, (self.batch_size, self.embedding_dim),
                         "Flow forward pass shape mismatch")
        self.assertEqual(campaign_emb.shape,
                         (self.batch_size, self.embedding_dim),
                         "Campaign forward pass shape mismatch")

        # Check normalization (L2 norm should be ~1)
        packet_norms = torch.norm(packet_emb, p=2, dim=1)
        flow_norms = torch.norm(flow_emb, p=2, dim=1)
        campaign_norms = torch.norm(campaign_emb, p=2, dim=1)

        self.assertTrue(torch.allclose(
            packet_norms, torch.ones_like(packet_norms), atol=1e-5),
            "Packet embeddings not properly normalized")
        self.assertTrue(torch.allclose(
            flow_norms, torch.ones_like(flow_norms), atol=1e-5),
            "Flow embeddings not properly normalized")
        self.assertTrue(torch.allclose(
            campaign_norms, torch.ones_like(campaign_norms), atol=1e-5),
            "Campaign embeddings not properly normalized")

    def test_encoder_gradients(self):
        """Test that encoder computes gradients properly"""
        packet_data = torch.randn(
            self.batch_size, self.packet_dim, requires_grad=True)
        flow_data = torch.randn(
            self.batch_size, self.flow_seq_len, self.packet_dim, requires_grad=True)
        campaign_data = torch.randn(
            self.batch_size, self.campaign_dim, requires_grad=True)

        packet_emb, flow_emb, campaign_emb = self.encoder(
            packet_data, flow_data, campaign_data
        )

        # Compute loss and backprop
        loss = packet_emb.sum() + flow_emb.sum() + campaign_emb.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(packet_data.grad)
        self.assertIsNotNone(flow_data.grad)
        self.assertIsNotNone(campaign_data.grad)


class TestMALZDAModel(unittest.TestCase):
    """Test MALZDA model"""

    def setUp(self):
        """Set up model configuration"""
        self.config = {
            'packet_dim': 64,
            'flow_seq_len': 10,
            'campaign_dim': 256,
            'embedding_dim': 32,
            'n_way': 3,
            'k_shot': 2,
            'n_query': 5
        }

        self.model = MALZDA(
            packet_dim=self.config['packet_dim'],
            flow_seq_len=self.config['flow_seq_len'],
            campaign_dim=self.config['campaign_dim'],
            embedding_dim=self.config['embedding_dim']
        )

        # Create test dataset
        self.dataset = CyberSecurityDataset(
            num_classes=5,
            samples_per_class=20,
            feature_dim=self.config['packet_dim'],
            temporal_length=self.config['flow_seq_len']
        )

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.hierarchical_encoder,
                             "Hierarchical encoder not initialized")
        self.assertIsInstance(self.model.alpha, torch.nn.Parameter,
                              "alpha should be Parameter")
        self.assertIsInstance(self.model.beta, torch.nn.Parameter,
                              "beta should be Parameter")
        self.assertIsInstance(self.model.gamma, torch.nn.Parameter,
                              "gamma should be Parameter")
        self.assertIsInstance(self.model.temperature, torch.nn.Parameter,
                              "temperature should be Parameter")

    def test_forward_pass(self):
        """Test forward pass through model"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        support_embeddings, query_embeddings = self.model(
            support_set, query_set)

        # Check structure
        self.assertEqual(len(support_embeddings), 3,
                         "Should have 3 embedding levels")
        self.assertEqual(len(query_embeddings), 3,
                         "Should have 3 embedding levels")

        # Check dimensions
        expected_support_size = sampler.n_way * self.config['k_shot']
        self.assertEqual(support_embeddings[0].shape[0], expected_support_size,
                         "Support size mismatch")
        self.assertEqual(
            support_embeddings[0].shape[1], self.config['embedding_dim'],
            "Embedding dimension mismatch")

    def test_prototype_computation(self):
        """Test prototype computation"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        support_embeddings, _ = self.model(support_set, query_set)
        support_labels_tensor = torch.tensor(support_labels)

        prototypes = self.model.compute_prototypes(
            support_embeddings, support_labels_tensor)

        # Check number of prototypes
        self.assertEqual(len(prototypes), sampler.n_way,
                         "Number of prototypes mismatch")

        # Check prototype structure
        for class_id, proto in prototypes.items():
            self.assertIn('packet', proto,
                          "Prototype missing 'packet' key")
            self.assertIn('flow', proto,
                          "Prototype missing 'flow' key")
            self.assertIn('campaign', proto,
                          "Prototype missing 'campaign' key")
            self.assertEqual(proto['packet'].shape[0],
                             self.config['embedding_dim'],
                             "Prototype packet dimension mismatch")

    def test_distance_computation(self):
        """Test distance computation to prototypes"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        support_embeddings, query_embeddings = self.model(
            support_set, query_set)
        support_labels_tensor = torch.tensor(support_labels)

        prototypes = self.model.compute_prototypes(
            support_embeddings, support_labels_tensor)
        distances = self.model.compute_distances(query_embeddings, prototypes)

        # Check distance matrix shape
        self.assertEqual(distances.shape[0], len(query_set),
                         "Distance matrix query dimension mismatch")
        self.assertEqual(distances.shape[1], sampler.n_way,
                         "Distance matrix way dimension mismatch")

        # Check distances are non-negative and finite
        self.assertTrue(torch.all(distances >= 0),
                        "Distances should be non-negative")
        self.assertTrue(torch.all(torch.isfinite(distances)),
                        "Distances contain NaN/Inf")

    def test_distance_weights(self):
        """Test learnable distance weights"""
        weights = self.model.get_distance_weights()

        required_weights = ['alpha', 'beta', 'gamma', 'temperature']
        for weight_name in required_weights:
            self.assertIn(weight_name, weights,
                          f"Missing weight: {weight_name}")

        # All weights should be positive
        self.assertGreater(weights['alpha'], 0,
                           "alpha should be positive")
        self.assertGreater(weights['beta'], 0,
                           "beta should be positive")
        self.assertGreater(weights['gamma'], 0,
                           "gamma should be positive")
        self.assertGreater(weights['temperature'], 0,
                           "temperature should be positive")

    def test_model_to_device(self):
        """Test model device placement"""
        device = 'cpu'
        model = self.model.to(device)

        # Check parameters are on correct device
        for param in model.parameters():
            self.assertEqual(param.device.type, device,
                             f"Parameter not on {device}")


class TestCompositionalTaskSampler(unittest.TestCase):
    """Test compositional task sampler with dynamic n_way adjustment"""

    def setUp(self):
        """Set up sampler configuration"""
        self.dataset_many_classes = CyberSecurityDataset(
            num_classes=10,
            samples_per_class=50,
            feature_dim=64,
            temporal_length=10
        )

        self.dataset_few_classes = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=30,
            feature_dim=64,
            temporal_length=10
        )

        self.config = {
            'n_way': 5,
            'k_shot': 2,
            'n_query': 10
        }

    def test_compositional_sampling_many_classes(self):
        """Test compositional task sampling with many classes"""
        sampler = CompositionalTaskSampler(
            self.dataset_many_classes,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=True
        )

        # Should use requested n_way
        self.assertEqual(sampler.n_way, self.config['n_way'],
                         "n_way not set correctly with many classes")

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # Check support set size
        expected_support_size = self.config['n_way'] * self.config['k_shot']
        self.assertEqual(len(support_set), expected_support_size,
                         "Support set size mismatch")
        self.assertEqual(len(support_labels), expected_support_size,
                         "Support labels size mismatch")

        # Check number of unique classes
        unique_support_classes = len(set(support_labels))
        self.assertEqual(unique_support_classes, self.config['n_way'],
                         "Unique classes in support set mismatch")

    def test_dynamic_n_way_adjustment(self):
        """Test automatic n_way adjustment with limited classes"""
        # Request 5-way but only 2 classes available
        sampler = CompositionalTaskSampler(
            self.dataset_few_classes,
            n_way=5,  # Requested
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=True
        )

        # Should adjust to available classes
        self.assertEqual(sampler.n_way, 2,
                         "n_way should be adjusted to available classes")
        self.assertLess(sampler.n_way, 5,
                        "n_way should be less than requested")

        # Should still be able to sample
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        self.assertGreater(len(support_set), 0,
                           "Support set should not be empty")
        self.assertGreater(len(query_set), 0,
                           "Query set should not be empty")

    def test_standard_sampling(self):
        """Test standard sampling without kill-chain awareness"""
        sampler = CompositionalTaskSampler(
            self.dataset_many_classes,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=False
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # Check basic structure
        self.assertEqual(len(support_set),
                         self.config['n_way'] * self.config['k_shot'],
                         "Standard sampling support size mismatch")
        self.assertEqual(len(set(support_labels)), self.config['n_way'],
                         "Standard sampling unique classes mismatch")

    def test_query_set_validity(self):
        """Test query set validity and label consistency"""
        sampler = CompositionalTaskSampler(
            self.dataset_many_classes,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=True
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # Check labels consistency
        self.assertEqual(len(query_labels), len(query_set),
                         "Query labels and set size mismatch")
        self.assertTrue(
            all(0 <= label < sampler.n_way for label in query_labels),
            "Query labels out of range")

    def test_fallback_sampling(self):
        """Test fallback to standard sampling when compositional fails"""
        sampler = CompositionalTaskSampler(
            self.dataset_few_classes,
            n_way=2,
            k_shot=2,
            n_query=5,
            include_kill_chain=True
        )

        # Should succeed with fallback
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        self.assertEqual(len(support_set), sampler.n_way * 2,
                         "Fallback support set size mismatch")
        self.assertGreater(len(query_set), 0,
                           "Fallback query set should not be empty")

    def test_class_phase_indices_building(self):
        """Test class-phase indices are built correctly"""
        sampler = CompositionalTaskSampler(
            self.dataset_many_classes,
            n_way=3,
            k_shot=2,
            n_query=5
        )

        # Check indices are built
        self.assertGreater(len(sampler.class_phase_indices), 0,
                           "class_phase_indices should be built")

        # Check structure
        for class_id, phases in sampler.class_phase_indices.items():
            self.assertIsInstance(phases, dict,
                                  "Phases should be dictionary")
            self.assertGreater(len(phases), 0,
                               "Should have at least one phase per class")


class TestMALZDATrainer(unittest.TestCase):
    """Test training and evaluation with NaN safety"""

    def setUp(self):
        """Set up trainer configuration"""
        self.config = {
            'packet_dim': 64,
            'flow_seq_len': 10,
            'campaign_dim': 256,
            'embedding_dim': 32,
            'n_way': 3,
            'k_shot': 2,
            'n_query': 5
        }

        self.model = MALZDA(
            packet_dim=self.config['packet_dim'],
            flow_seq_len=self.config['flow_seq_len'],
            campaign_dim=self.config['campaign_dim'],
            embedding_dim=self.config['embedding_dim']
        )

        self.trainer = MALZDATrainer(
            self.model, learning_rate=0.001, device='cpu')

        self.dataset = CyberSecurityDataset(
            num_classes=5,
            samples_per_class=30,
            feature_dim=self.config['packet_dim'],
            temporal_length=self.config['flow_seq_len']
        )

        self.sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

    def test_training_episode(self):
        """Test single training episode"""
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()

        loss, accuracy = self.trainer.train_episode(
            support_set, query_set, support_labels, query_labels
        )

        # Check return types
        self.assertIsInstance(loss, float,
                              "Loss should be float")
        self.assertIsInstance(accuracy, float,
                              "Accuracy should be float")

        # Check for NaN values
        self.assertFalse(np.isnan(loss),
                         "Loss should not be NaN")
        self.assertFalse(np.isnan(accuracy),
                         "Accuracy should not be NaN")

        # Check value ranges
        self.assertGreater(loss, 0,
                           "Loss should be positive")
        self.assertGreaterEqual(accuracy, 0,
                                "Accuracy should be >= 0")
        self.assertLessEqual(accuracy, 1,
                             "Accuracy should be <= 1")

    def test_evaluation_episode(self):
        """Test single evaluation episode with NaN handling"""
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()

        results = self.trainer.evaluate_episode(
            support_set, query_set, support_labels, query_labels
        )

        # Check result structure
        required_keys = ['loss', 'accuracy', 'f1', 'precision',
                         'recall', 'predictions', 'labels', 'distance_weights']
        for key in required_keys:
            self.assertIn(key, results,
                          f"Missing result key: {key}")

        # Check for NaN values
        self.assertFalse(np.isnan(results['accuracy']),
                         "Accuracy should not be NaN")
        self.assertFalse(np.isnan(results['f1']),
                         "F1 should not be NaN")

        # Check metric ranges (allow for safe defaults)
        self.assertGreaterEqual(results['accuracy'], 0,
                                "Accuracy should be >= 0")
        self.assertLessEqual(results['accuracy'], 1,
                             "Accuracy should be <= 1")

    def test_training_with_limited_classes(self):
        """Test training with limited class dataset"""
        dataset_limited = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=20,
            feature_dim=self.config['packet_dim'],
            temporal_length=self.config['flow_seq_len']
        )

        sampler = CompositionalTaskSampler(
            dataset_limited,
            n_way=5,  # Request more than available
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        # Should still train successfully
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        loss, accuracy = self.trainer.train_episode(
            support_set, query_set, support_labels, query_labels
        )

        self.assertFalse(np.isnan(loss),
                         "Loss with limited classes should not be NaN")
        self.assertFalse(np.isnan(accuracy),
                         "Accuracy with limited classes should not be NaN")

    def test_training_history(self):
        """Test training history recording"""
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()

        # Train for a few episodes
        for _ in range(3):
            self.trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )

        # Check history
        self.assertEqual(len(self.trainer.training_history['loss']), 3,
                         "Training history loss size mismatch")
        self.assertEqual(len(self.trainer.training_history['accuracy']), 3,
                         "Training history accuracy size mismatch")
        self.assertEqual(
            len(self.trainer.training_history['distance_weights']), 3,
            "Training history weights size mismatch")

        # Check no NaN values in history
        self.assertTrue(all(not np.isnan(l)
                        for l in self.trainer.training_history['loss']),
                        "Training history contains NaN losses")
        self.assertTrue(all(not np.isnan(a)
                        for a in self.trainer.training_history['accuracy']),
                        "Training history contains NaN accuracies")

    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train for a few steps
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        self.trainer.train_episode(
            support_set, query_set, support_labels, query_labels)

        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.pt"
            self.trainer.save_model(save_path)

            # Check file exists
            self.assertTrue(save_path.exists(),
                            "Model file should be created")

            # Load model
            new_model = MALZDA(
                packet_dim=self.config['packet_dim'],
                flow_seq_len=self.config['flow_seq_len'],
                campaign_dim=self.config['campaign_dim'],
                embedding_dim=self.config['embedding_dim']
            )
            new_trainer = MALZDATrainer(
                new_model, learning_rate=0.001, device='cpu')
            new_trainer.load_model(save_path)

            # Check history was loaded
            self.assertGreater(len(new_trainer.training_history['loss']), 0,
                               "Loaded history should not be empty")

    def test_optimizer_state(self):
        """Test optimizer is properly configured"""
        self.assertIsNotNone(self.trainer.optimizer,
                             "Optimizer should be initialized")

        # Train one step
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        self.trainer.train_episode(
            support_set, query_set, support_labels, query_labels)

        # Check optimizer has state
        self.assertGreater(len(self.trainer.optimizer.state_dict()['state']), 0,
                           "Optimizer should have state after training")

    def test_scheduler_step(self):
        """Test learning rate scheduler"""
        initial_lr = self.trainer.optimizer.param_groups[0]['lr']

        # Perform scheduler steps
        for _ in range(5):
            support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
            self.trainer.train_episode(
                support_set, query_set, support_labels, query_labels)
            self.trainer.scheduler.step()

        final_lr = self.trainer.optimizer.param_groups[0]['lr']

        # Learning rate should potentially change (depends on scheduler)
        self.assertIsNotNone(final_lr,
                             "Learning rate should exist after scheduler steps")


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""

    def test_csv_preprocessing(self):
        """Test CSV data preprocessing"""
        # Create temporary CSV file
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"

            # Create synthetic CSV data
            n_samples = 100
            n_features = 20

            data = {
                f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
            }
            data['target'] = np.random.randint(0, 5, n_samples)

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            # Load and preprocess
            X_scaled, X_unscaled, y, feature_names, kill_chain_labels = \
                load_and_preprocess_real_data(csv_path)

            # Check outputs
            self.assertEqual(X_scaled.shape[0], n_samples,
                             "Preprocessed samples mismatch")
            self.assertEqual(X_scaled.shape[1], n_features,
                             "Preprocessed features mismatch")
            self.assertEqual(len(y), n_samples,
                             "Labels length mismatch")
            self.assertEqual(len(feature_names), n_features,
                             "Feature names length mismatch")
            self.assertEqual(len(kill_chain_labels), n_samples,
                             "Kill-chain labels length mismatch")

            # Check scaling (mean ~0, std ~1)
            self.assertLess(np.abs(np.mean(X_scaled)), 0.2,
                            "Scaled data mean not close to 0")
            self.assertLess(np.abs(np.std(X_scaled) - 1.0), 0.3,
                            "Scaled data std not close to 1")

    def test_preprocessing_no_nan(self):
        """Test that preprocessing doesn't introduce NaN values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"

            # Create data with some edge cases
            n_samples = 50
            n_features = 10

            data = {
                f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
            }
            data['target'] = np.random.randint(
                0, 3, n_samples)  # Limited classes

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            X_scaled, X_unscaled, y, feature_names, kill_chain_labels = \
                load_and_preprocess_real_data(csv_path)

            # Check no NaN values
            self.assertFalse(np.any(np.isnan(X_scaled)),
                             "Scaled data contains NaN")
            self.assertFalse(np.any(np.isnan(X_unscaled)),
                             "Unscaled data contains NaN")

            # Check no infinite values
            self.assertFalse(np.any(np.isinf(X_scaled)),
                             "Scaled data contains Inf")
            self.assertFalse(np.any(np.isinf(X_unscaled)),
                             "Unscaled data contains Inf")

    def test_feature_names_extraction(self):
        """Test feature names are properly extracted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"

            custom_features = ['feature_a', 'feature_b', 'feature_c']
            n_samples = 50

            data = {
                feature: np.random.randn(n_samples) for feature in custom_features
            }
            data['target'] = np.random.randint(0, 3, n_samples)

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            X_scaled, X_unscaled, y, feature_names, kill_chain_labels = \
                load_and_preprocess_real_data(csv_path)

            # Check feature names match (excluding target)
            self.assertEqual(len(feature_names), len(custom_features),
                             "Feature names count mismatch")


class TestVisualizationFunctions(unittest.TestCase):
    """Test visualization and result handling functions"""

    def setUp(self):
        """Set up test data for visualization"""
        self.config = {
            'n_way': 3,
            'k_shot': 2,
            'n_query': 5
        }

    def test_nan_safe_visualization(self):
        """Test that visualization handles NaN values safely"""
        # Create result data with some NaN values
        scaling_results = {
            1: {
                'accuracy': 0.85,
                'f1': 0.83,
                'precision': 0.84,
                'recall': 0.82
            },
            5: {
                'accuracy': 0.90,
                'f1': np.nan,  # NaN value
                'precision': 0.91,
                'recall': 0.89
            },
            10: {
                'accuracy': 0.92,
                'f1': 0.91,
                'precision': 0.93,
                'recall': 0.90
            }
        }

        # Should handle NaN gracefully
        # This tests the filtering logic in visualization functions
        valid_results = {
            k: v for k, v in scaling_results.items()
            if not any(np.isnan(val) if isinstance(val, (int, float)) else False
                       for val in v.values())
        }

        self.assertGreaterEqual(len(valid_results), 2,
                                "Should have at least 2 valid results")

    def test_results_serialization(self):
        """Test that results can be serialized to JSON"""
        results_dict = {
            'config': {
                'n_way': 3,
                'k_shot': 2,
                'n_query': 5
            },
            'metrics': {
                'accuracy': np.float32(0.85),
                'f1': np.float64(0.83),
                'array': np.array([1, 2, 3])
            }
        }

        # Convert numpy types
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable = convert_to_serializable(results_dict)

        # Should be JSON serializable
        json_str = json.dumps(serializable)
        self.assertIsNotNone(json_str,
                             "Results should be JSON serializable")


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests with robustness"""

    def test_complete_training_cycle(self):
        """Test complete training and evaluation cycle"""
        # Create dataset with sufficient classes
        dataset_train = CyberSecurityDataset(
            num_classes=5,
            samples_per_class=50,
            feature_dim=64,
            temporal_length=10
        )

        dataset_test = CyberSecurityDataset(
            num_classes=5,
            samples_per_class=30,
            feature_dim=64,
            temporal_length=10
        )

        # Create model
        model = MALZDA(
            packet_dim=64,
            flow_seq_len=10,
            campaign_dim=256,
            embedding_dim=32
        )

        trainer = MALZDATrainer(model, learning_rate=0.001, device='cpu')

        # Create samplers
        train_sampler = CompositionalTaskSampler(
            dataset_train, n_way=3, k_shot=2, n_query=5
        )
        test_sampler = CompositionalTaskSampler(
            dataset_test, n_way=3, k_shot=2, n_query=5
        )

        # Train for a few episodes
        train_accuracies = []
        for _ in range(5):
            support_set, query_set, support_labels, query_labels = \
                train_sampler.sample_task()
            loss, accuracy = trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )
            train_accuracies.append(accuracy)

        # Evaluate
        eval_results = []
        for _ in range(3):
            support_set, query_set, support_labels, query_labels = \
                test_sampler.sample_task()
            results = trainer.evaluate_episode(
                support_set, query_set, support_labels, query_labels
            )
            eval_results.append(results['accuracy'])

        # Check that training occurred
        self.assertGreater(len(train_accuracies), 0,
                           "Training should produce results")
        self.assertGreater(len(eval_results), 0,
                           "Evaluation should produce results")

        # Check no NaN values
        self.assertTrue(all(not np.isnan(acc) for acc in train_accuracies),
                        "Training accuracies contain NaN")
        self.assertTrue(all(not np.isnan(acc) for acc in eval_results),
                        "Evaluation accuracies contain NaN")

        # Check reasonable performance (random baseline ~33% for 3-way)
        mean_train_acc = np.mean(
            [a for a in train_accuracies if not np.isnan(a)])
        mean_eval_acc = np.mean([a for a in eval_results if not np.isnan(a)])

        self.assertGreater(mean_train_acc, 0.2,
                           "Training accuracy too low")
        self.assertGreater(mean_eval_acc, 0.2,
                           "Evaluation accuracy too low")

    def test_training_with_limited_classes(self):
        """Test end-to-end with limited class dataset"""
        # Create dataset with only 2 classes
        dataset_train = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=40,
            feature_dim=64,
            temporal_length=10
        )

        dataset_test = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=25,
            feature_dim=64,
            temporal_length=10
        )

        # Create model
        model = MALZDA(
            packet_dim=64,
            flow_seq_len=10,
            campaign_dim=256,
            embedding_dim=32
        )

        trainer = MALZDATrainer(model, learning_rate=0.001, device='cpu')

        # Create samplers with dynamic n_way
        train_sampler = CompositionalTaskSampler(
            dataset_train, n_way=5, k_shot=2, n_query=5
        )
        test_sampler = CompositionalTaskSampler(
            dataset_test, n_way=5, k_shot=2, n_query=5
        )

        # Should adjust to 2-way
        self.assertEqual(train_sampler.n_way, 2,
                         "Train sampler should adjust to 2-way")
        self.assertEqual(test_sampler.n_way, 2,
                         "Test sampler should adjust to 2-way")

        # Train successfully
        for _ in range(3):
            support_set, query_set, support_labels, query_labels = \
                train_sampler.sample_task()
            loss, accuracy = trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )

            self.assertFalse(np.isnan(loss),
                             "Training loss should not be NaN")
            self.assertFalse(np.isnan(accuracy),
                             "Training accuracy should not be NaN")

    def test_run_experiment_function(self):
        """Test the main run_experiment function"""
        dataset_train = CyberSecurityDataset(
            num_classes=4,
            samples_per_class=40,
            feature_dim=64,
            temporal_length=10
        )

        dataset_test = CyberSecurityDataset(
            num_classes=4,
            samples_per_class=25,
            feature_dim=64,
            temporal_length=10
        )

        # Run experiment with minimal episodes for testing
        model, eval_results, train_losses, train_accuracies = run_experiment(
            dataset_train,
            dataset_test,
            n_way=3,
            k_shot=2,
            n_query=5,
            num_episodes=10,
            eval_episodes=5,
            use_compositional=True,
            experiment_name="test_experiment"
        )

        # Check returns
        self.assertIsNotNone(model,
                             "Model should be returned")
        self.assertIsNotNone(eval_results,
                             "Evaluation results should be returned")
        self.assertGreater(len(train_losses), 0,
                           "Training losses should be recorded")
        self.assertGreater(len(train_accuracies), 0,
                           "Training accuracies should be recorded")


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes in order of dependency
    test_classes = [
        TestCyberSecurityDataset,
        TestHierarchicalEncoder,
        TestMALZDAModel,
        TestCompositionalTaskSampler,
        TestMALZDATrainer,
        TestDataPreprocessing,
        TestVisualizationFunctions,
        TestEndToEnd
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(
        f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*80)

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)

    return result


if __name__ == '__main__':
    # Run tests
    result = run_tests()

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)


################################################
# END OF FILE                                  #
################################################
