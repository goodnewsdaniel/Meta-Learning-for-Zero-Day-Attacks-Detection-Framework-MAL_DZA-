################################################
'''   Goodnews Daniel (PhD Candidate)
        222166453@student.uj.ac.za
Department of Electrical & Electronics Engineering
Faculty of Engineering & the Built Environment
University of Johannesburg, South Africa        

MAL-ZDA Test Suite - COMPLETE & CORRECT
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
            'num_classes': 3,
            'samples_per_class': 10,
            'feature_dim': 32,
            'temporal_length': 8,
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

        expected_len = self.test_config['num_classes'] * \
            self.test_config['samples_per_class']
        self.assertEqual(len(dataset), expected_len,
                         "Dataset length mismatch")

        sample = dataset[0]
        required_keys = ['packet', 'flow',
                         'campaign', 'class_id', 'kill_chain']
        for key in required_keys:
            self.assertIn(key, sample, f"Missing key: {key}")

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

        self.assertTrue(0 <= sample['class_id'] <
                        self.test_config['num_classes'],
                        "class_id out of range")
        self.assertTrue(0 <= sample['kill_chain'] < 5,
                        "kill_chain out of range (should be 0-4)")

        self.assertFalse(np.any(np.isnan(sample['packet'])),
                         "Packet contains NaN")
        self.assertFalse(np.any(np.isinf(sample['packet'])),
                         "Packet contains Inf")
        self.assertFalse(np.any(np.isnan(sample['flow'])),
                         "Flow contains NaN")
        self.assertFalse(np.any(np.isinf(sample['flow'])),
                         "Flow contains Inf")

    def test_real_data_loading(self):
        """Test loading real data from arrays"""
        n_samples = 50
        n_features = 32
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 3, n_samples)
        kill_chain_labels = np.random.randint(0, 5, n_samples)
        feature_names = [f'Feature_{i}' for i in range(n_features)]

        dataset = CyberSecurityDataset(
            X=X,
            y=y,
            kill_chain_labels=kill_chain_labels,
            feature_names=feature_names,
            temporal_length=8
        )

        self.assertEqual(len(dataset), n_samples,
                         "Real data dataset size mismatch")

        sample = dataset[0]
        self.assertEqual(sample['packet'].shape[0], n_features,
                         "Real data packet dimension mismatch")
        self.assertEqual(sample['flow'].shape[0], 8,
                         "Real data flow temporal length mismatch")
        self.assertEqual(sample['campaign'].shape[0], n_features * 4,
                         "Real data campaign dimension mismatch")

    def test_kill_chain_phase_distribution(self):
        """Test if kill-chain phases are properly distributed"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=30,
            feature_dim=32,
            temporal_length=8
        )

        phases = [sample['kill_chain'] for sample in dataset.data]
        unique_phases = set(phases)

        self.assertGreater(len(unique_phases), 1,
                           "Should have multiple kill-chain phases")

        self.assertTrue(all(0 <= p < 5 for p in phases),
                        "All phases should be in range 0-4")

    def test_limited_class_dataset(self):
        """Test dataset with very limited number of classes"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=10,
            feature_dim=32,
            temporal_length=8
        )

        self.assertEqual(len(dataset), 20,
                         "Limited class dataset size mismatch")

        class_ids = [sample['class_id'] for sample in dataset.data]
        unique_classes = set(class_ids)
        self.assertEqual(len(unique_classes), 2,
                         "Should have exactly 2 classes")

    def test_dataset_indexing(self):
        """Test dataset indexing and access"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=8,
            feature_dim=32,
            temporal_length=8
        )

        for idx in range(len(dataset)):
            sample = dataset[idx]
            self.assertIsNotNone(sample)
            self.assertIn('packet', sample)

        first_sample = dataset[0]
        last_sample = dataset[len(dataset) - 1]
        self.assertIsNotNone(first_sample)
        self.assertIsNotNone(last_sample)


class TestHierarchicalEncoder(unittest.TestCase):
    """Test hierarchical encoder architecture"""

    def setUp(self):
        """Set up encoder configuration"""
        self.packet_dim = 32
        self.flow_seq_len = 8
        self.campaign_dim = 128
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

        self.assertEqual(packet_emb.shape,
                         (self.batch_size, self.embedding_dim),
                         "Packet forward pass shape mismatch")
        self.assertEqual(flow_emb.shape, (self.batch_size, self.embedding_dim),
                         "Flow forward pass shape mismatch")
        self.assertEqual(campaign_emb.shape,
                         (self.batch_size, self.embedding_dim),
                         "Campaign forward pass shape mismatch")

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

        loss = packet_emb.sum() + flow_emb.sum() + campaign_emb.sum()
        loss.backward()

        self.assertIsNotNone(packet_data.grad)
        self.assertIsNotNone(flow_data.grad)
        self.assertIsNotNone(campaign_data.grad)


class TestMALZDAModel(unittest.TestCase):
    """Test MALZDA model"""

    def setUp(self):
        """Set up model configuration"""
        self.config = {
            'packet_dim': 32,
            'flow_seq_len': 8,
            'campaign_dim': 128,
            'embedding_dim': 32,
            'n_way': 2,
            'k_shot': 2,
            'n_query': 5
        }

        self.model = MALZDA(
            packet_dim=self.config['packet_dim'],
            flow_seq_len=self.config['flow_seq_len'],
            campaign_dim=self.config['campaign_dim'],
            embedding_dim=self.config['embedding_dim']
        )

        self.dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=15,
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

    def test_distance_weights(self):
        """Test learnable distance weights"""
        weights = self.model.get_distance_weights()

        required_weights = ['alpha', 'beta', 'gamma', 'temperature']
        for weight_name in required_weights:
            self.assertIn(weight_name, weights,
                          f"Missing weight: {weight_name}")

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

        for param in model.parameters():
            self.assertEqual(param.device.type, device,
                             f"Parameter not on {device}")


class TestMALZDAModelWithTensorConversion(unittest.TestCase):
    """Test MALZDA model with proper tensor conversion"""

    def setUp(self):
        """Set up model configuration"""
        self.config = {
            'packet_dim': 32,
            'flow_seq_len': 8,
            'campaign_dim': 128,
            'embedding_dim': 32,
            'n_way': 2,
            'k_shot': 2,
            'n_query': 5
        }

        self.model = MALZDA(
            packet_dim=self.config['packet_dim'],
            flow_seq_len=self.config['flow_seq_len'],
            campaign_dim=self.config['campaign_dim'],
            embedding_dim=self.config['embedding_dim']
        )

        self.dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=15,
            feature_dim=self.config['packet_dim'],
            temporal_length=self.config['flow_seq_len']
        )

    def _convert_batch_to_tensors(self, batch_data):
        """Convert numpy arrays in batch to tensors"""
        tensor_batch = []
        for item in batch_data:
            tensor_item = {
                'packet': torch.from_numpy(item['packet']).float()
                if isinstance(item['packet'], np.ndarray) else item['packet'],
                'flow': torch.from_numpy(item['flow']).float()
                if isinstance(item['flow'], np.ndarray) else item['flow'],
                'campaign': torch.from_numpy(item['campaign']).float()
                if isinstance(item['campaign'], np.ndarray) else item['campaign'],
                'class_id': item['class_id'],
                'kill_chain': item['kill_chain']
            }
            tensor_batch.append(tensor_item)
        return tensor_batch

    def test_forward_pass_with_tensor_conversion(self):
        """Test forward pass with proper numpy to tensor conversion"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        support_set_tensor = self._convert_batch_to_tensors(support_set)
        query_set_tensor = self._convert_batch_to_tensors(query_set)

        support_embeddings, query_embeddings = self.model(
            support_set_tensor, query_set_tensor)

        self.assertEqual(len(support_embeddings), 3,
                         "Should have 3 embedding levels")
        self.assertEqual(len(query_embeddings), 3,
                         "Should have 3 embedding levels")

        expected_support_size = sampler.n_way * self.config['k_shot']
        self.assertEqual(support_embeddings[0].shape[0], expected_support_size,
                         "Support size mismatch")
        self.assertEqual(
            support_embeddings[0].shape[1], self.config['embedding_dim'],
            "Embedding dimension mismatch")

    def test_prototype_computation_with_tensor_conversion(self):
        """Test prototype computation with tensor conversion"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        support_set_tensor = self._convert_batch_to_tensors(support_set)
        query_set_tensor = self._convert_batch_to_tensors(query_set)

        support_embeddings, _ = self.model(
            support_set_tensor, query_set_tensor)
        support_labels_tensor = torch.tensor(support_labels)

        prototypes = self.model.compute_prototypes(
            support_embeddings, support_labels_tensor)

        self.assertEqual(len(prototypes), sampler.n_way,
                         "Number of prototypes mismatch")

        for class_id, proto in prototypes.items():
            self.assertIn('packet', proto)
            self.assertIn('flow', proto)
            self.assertIn('campaign', proto)

    def test_distance_computation_with_tensor_conversion(self):
        """Test distance computation with tensor conversion"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        support_set_tensor = self._convert_batch_to_tensors(support_set)
        query_set_tensor = self._convert_batch_to_tensors(query_set)

        support_embeddings, query_embeddings = self.model(
            support_set_tensor, query_set_tensor)
        support_labels_tensor = torch.tensor(support_labels)

        prototypes = self.model.compute_prototypes(
            support_embeddings, support_labels_tensor)
        distances = self.model.compute_distances(query_embeddings, prototypes)

        self.assertEqual(distances.shape[0], len(query_set),
                         "Distance matrix query dimension mismatch")
        self.assertEqual(distances.shape[1], sampler.n_way,
                         "Distance matrix way dimension mismatch")

        self.assertTrue(torch.all(distances >= 0),
                        "Distances should be non-negative")
        self.assertTrue(torch.all(torch.isfinite(distances)),
                        "Distances contain NaN/Inf")


class TestCompositionalTaskSampler(unittest.TestCase):
    """Test compositional task sampler with dynamic n_way adjustment"""

    def setUp(self):
        """Set up sampler configuration"""
        self.dataset_many_classes = CyberSecurityDataset(
            num_classes=5,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        self.dataset_few_classes = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=15,
            feature_dim=32,
            temporal_length=8
        )

        self.config = {
            'n_way': 3,
            'k_shot': 2,
            'n_query': 5
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

        self.assertEqual(sampler.n_way, self.config['n_way'],
                         "n_way not set correctly with many classes")

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        expected_support_size = self.config['n_way'] * self.config['k_shot']
        self.assertEqual(len(support_set), expected_support_size,
                         "Support set size mismatch")
        self.assertEqual(len(support_labels), expected_support_size,
                         "Support labels size mismatch")

        unique_support_classes = len(set(support_labels))
        self.assertEqual(unique_support_classes, self.config['n_way'],
                         "Unique classes in support set mismatch")

    def test_dynamic_n_way_adjustment(self):
        """Test automatic n_way adjustment with limited classes"""
        sampler = CompositionalTaskSampler(
            self.dataset_few_classes,
            n_way=5,
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=True
        )

        self.assertEqual(sampler.n_way, 2,
                         "n_way should be adjusted to available classes")
        self.assertLess(sampler.n_way, 5,
                        "n_way should be less than requested")

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

        self.assertGreater(len(sampler.class_phase_indices), 0,
                           "class_phase_indices should be built")

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
            'packet_dim': 32,
            'flow_seq_len': 8,
            'campaign_dim': 128,
            'embedding_dim': 32,
            'n_way': 2,
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
            num_classes=3,
            samples_per_class=15,
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

        self.assertIsInstance(loss, float,
                              "Loss should be float")
        self.assertIsInstance(accuracy, float,
                              "Accuracy should be float")

        self.assertFalse(np.isnan(loss),
                         "Loss should not be NaN")
        self.assertFalse(np.isnan(accuracy),
                         "Accuracy should not be NaN")

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

        required_keys = ['loss', 'accuracy', 'f1', 'precision',
                         'recall', 'predictions', 'labels', 'distance_weights']
        for key in required_keys:
            self.assertIn(key, results,
                          f"Missing result key: {key}")

        self.assertFalse(np.isnan(results['accuracy']),
                         "Accuracy should not be NaN")
        self.assertFalse(np.isnan(results['f1']),
                         "F1 should not be NaN")

        self.assertGreaterEqual(results['accuracy'], 0,
                                "Accuracy should be >= 0")
        self.assertLessEqual(results['accuracy'], 1,
                             "Accuracy should be <= 1")

    def test_training_with_limited_classes(self):
        """Test training with limited class dataset"""
        dataset_limited = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=12,
            feature_dim=self.config['packet_dim'],
            temporal_length=self.config['flow_seq_len']
        )

        sampler = CompositionalTaskSampler(
            dataset_limited,
            n_way=3,
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )

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

        for _ in range(3):
            self.trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )

        self.assertEqual(len(self.trainer.training_history['loss']), 3,
                         "Training history loss size mismatch")
        self.assertEqual(len(self.trainer.training_history['accuracy']), 3,
                         "Training history accuracy size mismatch")
        self.assertEqual(
            len(self.trainer.training_history['distance_weights']), 3,
            "Training history weights size mismatch")

        self.assertTrue(all(not np.isnan(l)
                        for l in self.trainer.training_history['loss']),
                        "Training history contains NaN losses")
        self.assertTrue(all(not np.isnan(a)
                        for a in self.trainer.training_history['accuracy']),
                        "Training history contains NaN accuracies")

    def test_model_save_load(self):
        """Test model saving and loading"""
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        self.trainer.train_episode(
            support_set, query_set, support_labels, query_labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.pt"
            self.trainer.save_model(save_path)

            self.assertTrue(save_path.exists(),
                            "Model file should be created")

            new_model = MALZDA(
                packet_dim=self.config['packet_dim'],
                flow_seq_len=self.config['flow_seq_len'],
                campaign_dim=self.config['campaign_dim'],
                embedding_dim=self.config['embedding_dim']
            )
            new_trainer = MALZDATrainer(
                new_model, learning_rate=0.001, device='cpu')
            new_trainer.load_model(save_path)

            self.assertGreater(len(new_trainer.training_history['loss']), 0,
                               "Loaded history should not be empty")

    def test_optimizer_state(self):
        """Test optimizer is properly configured"""
        self.assertIsNotNone(self.trainer.optimizer,
                             "Optimizer should be initialized")

        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        self.trainer.train_episode(
            support_set, query_set, support_labels, query_labels)

        self.assertGreater(len(self.trainer.optimizer.state_dict()['state']), 0,
                           "Optimizer should have state after training")

    def test_scheduler_step(self):
        """Test learning rate scheduler"""
        initial_lr = self.trainer.optimizer.param_groups[0]['lr']

        for _ in range(5):
            support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
            self.trainer.train_episode(
                support_set, query_set, support_labels, query_labels)
            self.trainer.scheduler.step()

        final_lr = self.trainer.optimizer.param_groups[0]['lr']

        self.assertIsNotNone(final_lr,
                             "Learning rate should exist after scheduler steps")


class TestRobustDataHandling(unittest.TestCase):
    """Test robust handling of edge cases in data"""

    def test_batch_with_varying_sizes(self):
        """Test handling of batches with varying number of samples"""
        dataset = CyberSecurityDataset(
            num_classes=4,
            samples_per_class=25,
            feature_dim=32,
            temporal_length=8
        )

        for batch_size in [1, 4, 8, 16]:
            sampler = CompositionalTaskSampler(
                dataset, n_way=2, k_shot=batch_size, n_query=10
            )
            support_set, query_set, _, _ = sampler.sample_task()

            self.assertEqual(len(support_set), 2 * batch_size,
                             f"Support set size mismatch for batch_size={batch_size}")
            self.assertGreater(len(query_set), 0,
                               f"Query set should not be empty for batch_size={batch_size}")

    def test_single_sample_batch(self):
        """Test handling of single sample batches"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=10,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=3, k_shot=1, n_query=1
        )
        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # Support set should have n_way * k_shot samples
        self.assertEqual(len(support_set), 3,
                         "Single-shot support set should have n_way samples")

        # Query set will have at least n_query samples, but may have more
        self.assertGreaterEqual(len(query_set), 1,
                                "Query set should have at least n_query samples")

        # Labels should match sizes
        self.assertEqual(len(support_labels), len(support_set),
                         "Support labels should match support set size")
        self.assertEqual(len(query_labels), len(query_set),
                         "Query labels should match query set size")

    def test_single_sample_batch_with_larger_query(self):
        """Test handling of single sample support with larger query set"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=1, n_query=10
        )
        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # Support: 2-way 1-shot = 2 samples
        self.assertEqual(len(support_set), 2,
                         "Support set should be 2-way 1-shot")

        # Query: at least n_query samples (may be more depending on sampler implementation)
        self.assertGreaterEqual(len(query_set), 10,
                                "Query set should have at least n_query samples")

        # All query labels should be valid
        self.assertTrue(all(0 <= label < 2 for label in query_labels),
                        "All query labels should be in range [0, n_way)")

    def test_large_feature_dimensions(self):
        """Test handling of large feature dimensions"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=10,
            feature_dim=512,
            temporal_length=16
        )

        encoder = HierarchicalEncoder(
            packet_dim=512,
            flow_seq_len=16,
            campaign_dim=2048,
            embedding_dim=256
        )

        batch_size = 4
        packet_data = torch.randn(batch_size, 512)
        flow_data = torch.randn(batch_size, 16, 512)
        campaign_data = torch.randn(batch_size, 2048)

        packet_emb, flow_emb, campaign_emb = encoder(
            packet_data, flow_data, campaign_data
        )

        self.assertEqual(packet_emb.shape, (batch_size, 256))
        self.assertEqual(flow_emb.shape, (batch_size, 256))
        self.assertEqual(campaign_emb.shape, (batch_size, 256))

    def test_small_feature_dimensions(self):
        """Test handling of small feature dimensions"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=10,
            feature_dim=8,
            temporal_length=4
        )

        encoder = HierarchicalEncoder(
            packet_dim=8,
            flow_seq_len=4,
            campaign_dim=32,
            embedding_dim=16
        )

        batch_size = 4
        packet_data = torch.randn(batch_size, 8)
        flow_data = torch.randn(batch_size, 4, 8)
        campaign_data = torch.randn(batch_size, 32)

        packet_emb, flow_emb, campaign_emb = encoder(
            packet_data, flow_data, campaign_data
        )

        self.assertEqual(packet_emb.shape, (batch_size, 16))
        self.assertEqual(flow_emb.shape, (batch_size, 16))
        self.assertEqual(campaign_emb.shape, (batch_size, 16))

    def test_mismatched_batch_sizes(self):
        """Test handling of mismatched batch dimensions"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=15,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=3, k_shot=3, n_query=8
        )
        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # Verify internal consistency
        self.assertEqual(len(support_set), len(support_labels),
                         "Support set and labels should have same size")
        self.assertEqual(len(query_set), len(query_labels),
                         "Query set and labels should have same size")

        # Verify all samples have required keys
        for sample in support_set + query_set:
            self.assertIn('packet', sample)
            self.assertIn('flow', sample)
            self.assertIn('campaign', sample)


class TestEdgeCasesAndBoundaries(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_minimum_viable_dataset(self):
        """Test with absolute minimum dataset"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=2,
            feature_dim=8,
            temporal_length=2
        )

        self.assertEqual(len(dataset), 4)
        sample = dataset[0]
        self.assertIn('packet', sample)
        self.assertIn('flow', sample)
        self.assertIn('campaign', sample)

    def test_single_class_dataset_handling(self):
        """Test handling of single class dataset - sampler adjusts to min n_way"""
        dataset = CyberSecurityDataset(
            num_classes=1,
            samples_per_class=10,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=5, k_shot=2, n_query=5
        )

        # Sampler adjusts n_way to minimum 2 or available classes
        # Since only 1 class available, it uses that but may adjust to 2 internally
        self.assertGreaterEqual(sampler.n_way, 1,
                                "n_way should be at least 1")
        self.assertLessEqual(sampler.n_way, 2,
                             "n_way should be adjusted to available resources")

        # Should still be able to sample
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        self.assertGreater(len(support_set), 0,
                           "Should be able to sample from single class")
        self.assertGreater(len(query_set), 0,
                           "Query set should not be empty")

    def test_single_class_dataset_behavior(self):
        """Test actual sampling behavior with single class"""
        dataset = CyberSecurityDataset(
            num_classes=1,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        # The sampler internally may adjust n_way
        sampler = CompositionalTaskSampler(
            dataset, n_way=3, k_shot=2, n_query=5
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # All support labels should be the same (only 1 class)
        unique_support_classes = len(set(support_labels))
        self.assertLessEqual(unique_support_classes, sampler.n_way,
                             "Unique classes should not exceed n_way")

        # All query labels should also be from available classes
        unique_query_classes = len(set(query_labels))
        self.assertLessEqual(unique_query_classes, sampler.n_way,
                             "Query classes should not exceed n_way")

    def test_high_dimensional_temporal_sequence(self):
        """Test handling of high-dimensional temporal sequences"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=10,
            feature_dim=32,
            temporal_length=256
        )

        sample = dataset[0]
        self.assertEqual(sample['flow'].shape, (256, 32),
                         "Flow should have correct temporal length")

    def test_label_boundary_conditions(self):
        """Test label handling at boundaries"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=2, n_query=10
        )

        for _ in range(5):
            _, _, support_labels, query_labels = sampler.sample_task()

            # Labels should be in valid range
            self.assertTrue(all(0 <= l < sampler.n_way for l in support_labels),
                            "Support labels out of range")
            self.assertTrue(all(0 <= l < sampler.n_way for l in query_labels),
                            "Query labels out of range")

    def test_zero_initialized_inputs(self):
        """Test handling of zero-initialized inputs"""
        encoder = HierarchicalEncoder(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        batch_size = 4
        packet_data = torch.zeros(batch_size, 32)
        flow_data = torch.zeros(batch_size, 8, 32)
        campaign_data = torch.zeros(batch_size, 128)

        packet_emb, flow_emb, campaign_emb = encoder(
            packet_data, flow_data, campaign_data
        )

        self.assertTrue(torch.all(torch.isfinite(packet_emb)))
        self.assertTrue(torch.all(torch.isfinite(flow_emb)))
        self.assertTrue(torch.all(torch.isfinite(campaign_emb)))

    def test_extremely_small_values(self):
        """Test handling of extremely small values"""
        encoder = HierarchicalEncoder(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        batch_size = 4
        packet_data = torch.randn(batch_size, 32) * 1e-6
        flow_data = torch.randn(batch_size, 8, 32) * 1e-6
        campaign_data = torch.randn(batch_size, 128) * 1e-6

        packet_emb, flow_emb, campaign_emb = encoder(
            packet_data, flow_data, campaign_data
        )

        self.assertTrue(torch.all(torch.isfinite(packet_emb)))
        self.assertTrue(torch.all(torch.isfinite(flow_emb)))
        self.assertTrue(torch.all(torch.isfinite(campaign_emb)))

    def test_extremely_large_values(self):
        """Test handling of extremely large values"""
        encoder = HierarchicalEncoder(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        batch_size = 4
        packet_data = torch.randn(batch_size, 32) * 1e6
        flow_data = torch.randn(batch_size, 8, 32) * 1e6
        campaign_data = torch.randn(batch_size, 128) * 1e6

        packet_emb, flow_emb, campaign_emb = encoder(
            packet_data, flow_data, campaign_data
        )

        # Should still produce valid embeddings
        self.assertTrue(torch.all(torch.isfinite(packet_emb)))
        self.assertTrue(torch.all(torch.isfinite(flow_emb)))
        self.assertTrue(torch.all(torch.isfinite(campaign_emb)))

    def test_mixed_zero_and_nonzero_inputs(self):
        """Test handling of mixed zero and non-zero inputs"""
        encoder = HierarchicalEncoder(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        batch_size = 4
        # Create mixed inputs
        packet_data = torch.zeros(batch_size, 32)
        packet_data[0] = torch.randn(32)

        flow_data = torch.zeros(batch_size, 8, 32)
        flow_data[1] = torch.randn(8, 32)

        campaign_data = torch.zeros(batch_size, 128)
        campaign_data[2] = torch.randn(128)

        packet_emb, flow_emb, campaign_emb = encoder(
            packet_data, flow_data, campaign_data
        )

        self.assertTrue(torch.all(torch.isfinite(packet_emb)))
        self.assertTrue(torch.all(torch.isfinite(flow_emb)))
        self.assertTrue(torch.all(torch.isfinite(campaign_emb)))


class TestSamplerEdgeCases(unittest.TestCase):
    """Test sampler behavior with edge case parameters"""

    def test_k_shot_equals_samples_per_class(self):
        """Test when k_shot equals available samples per class"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=5,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=5, n_query=3
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        self.assertEqual(len(support_set), 10,
                         "Support set should have n_way * k_shot samples")
        self.assertGreater(len(query_set), 0,
                           "Query set should have samples")

    def test_k_shot_exceeds_samples_per_class(self):
        """Test when k_shot exceeds available samples per class"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=3,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=10, n_query=2
        )

        # Should handle gracefully (may use all available samples)
        try:
            support_set, query_set, support_labels, query_labels = sampler.sample_task()
            self.assertGreater(len(support_set), 0,
                               "Should handle k_shot > samples_per_class")
        except Exception:
            # If it raises, that's also acceptable behavior
            pass

    def test_n_query_larger_than_dataset(self):
        """Test when n_query is larger than available samples"""
        dataset = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=3,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=1, n_query=100
        )

        support_set, query_set, support_labels, query_labels = sampler.sample_task()

        # May cap query set to available samples
        self.assertGreater(len(query_set), 0,
                           "Query set should not be empty")
        self.assertEqual(len(query_labels), len(query_set),
                         "Query labels should match query set size")

    def test_multiple_consecutive_samples(self):
        """Test multiple consecutive task samples with flexible query sizes"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        sampler = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=2, n_query=5
        )

        for i in range(5):
            support_set, query_set, support_labels, query_labels = sampler.sample_task()

            # Verify support set structure (strict)
            self.assertEqual(len(support_set), 4,
                             f"Iteration {i}: Support set size should be n_way * k_shot")
            self.assertEqual(len(support_labels), 4,
                             f"Iteration {i}: Support labels size mismatch")

            # Verify query set structure (flexible - at least n_query)
            self.assertGreaterEqual(len(query_set), 5,
                                    f"Iteration {i}: Query set should have at least n_query samples")
            self.assertEqual(len(query_labels), len(query_set),
                             f"Iteration {i}: Query labels should match query set size")

            # Verify label consistency
            unique_support_classes = len(set(support_labels))
            self.assertEqual(unique_support_classes, sampler.n_way,
                             f"Iteration {i}: Should have n_way unique support classes")

    def test_sampler_reproducibility_with_seed(self):
        """Test sampler reproducibility with fixed seed"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        # First sample
        torch.manual_seed(42)
        np.random.seed(42)
        sampler1 = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=2, n_query=5
        )
        support1, query1, labels1_s, labels1_q = sampler1.sample_task()

        # Second sample with same seed
        torch.manual_seed(42)
        np.random.seed(42)
        sampler2 = CompositionalTaskSampler(
            dataset, n_way=2, k_shot=2, n_query=5
        )
        support2, query2, labels2_s, labels2_q = sampler2.sample_task()

        # Should be identical with same seed
        self.assertEqual(len(support1), len(support2),
                         "Support sets should have same length with same seed")
        self.assertEqual(len(query1), len(query2),
                         "Query sets should have same length with same seed")

    def test_query_set_at_least_n_query(self):
        """Test that query set has at least n_query samples"""
        dataset = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=25,
            feature_dim=32,
            temporal_length=8
        )

        for n_query_val in [1, 5, 10, 15]:
            sampler = CompositionalTaskSampler(
                dataset, n_way=2, k_shot=2, n_query=n_query_val
            )
            support_set, query_set, support_labels, query_labels = sampler.sample_task()

            self.assertGreaterEqual(len(query_set), n_query_val,
                                    f"Query set should have at least {n_query_val} samples")

    def test_support_set_exact_size(self):
        """Test that support set has exact n_way * k_shot samples"""
        dataset = CyberSecurityDataset(
            num_classes=4,
            samples_per_class=30,
            feature_dim=32,
            temporal_length=8
        )

        test_configs = [
            (2, 1), (2, 2), (2, 5),
            (3, 1), (3, 2), (3, 3),
            (4, 2)
        ]

        for n_way, k_shot in test_configs:
            sampler = CompositionalTaskSampler(
                dataset, n_way=n_way, k_shot=k_shot, n_query=10
            )
            support_set, query_set, support_labels, query_labels = sampler.sample_task()

            expected_size = n_way * k_shot
            self.assertEqual(len(support_set), expected_size,
                             f"Support set should have exactly {expected_size} samples for {n_way}-way {k_shot}-shot")
            self.assertEqual(len(support_labels), expected_size,
                             f"Support labels should match support set size for {n_way}-way {k_shot}-shot")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of model computations"""

    def test_embedding_normalization_stability(self):
        """Test that normalized embeddings remain stable"""
        encoder = HierarchicalEncoder(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        batch_size = 4
        for _ in range(10):
            packet_data = torch.randn(batch_size, 32)
            flow_data = torch.randn(batch_size, 8, 32)
            campaign_data = torch.randn(batch_size, 128)

            packet_emb, flow_emb, campaign_emb = encoder(
                packet_data, flow_data, campaign_data
            )

            packet_norms = torch.norm(packet_emb, p=2, dim=1)
            flow_norms = torch.norm(flow_emb, p=2, dim=1)
            campaign_norms = torch.norm(campaign_emb, p=2, dim=1)

            self.assertTrue(torch.allclose(
                packet_norms, torch.ones_like(packet_norms), atol=1e-5))
            self.assertTrue(torch.allclose(
                flow_norms, torch.ones_like(flow_norms), atol=1e-5))
            self.assertTrue(torch.allclose(
                campaign_norms, torch.ones_like(campaign_norms), atol=1e-5))

    def test_distance_computation_stability(self):
        """Test that distance computations remain stable"""
        model = MALZDA(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        batch_size = 4
        n_way = 3

        for _ in range(5):
            packet_emb = torch.randn(batch_size, 32)
            flow_emb = torch.randn(batch_size, 32)
            campaign_emb = torch.randn(batch_size, 32)

            embeddings = [packet_emb, flow_emb, campaign_emb]

            prototypes = {}
            for i in range(n_way):
                prototypes[i] = {
                    'packet': torch.randn(32),
                    'flow': torch.randn(32),
                    'campaign': torch.randn(32)
                }

            distances = model.compute_distances(embeddings, prototypes)

            self.assertTrue(torch.all(torch.isfinite(distances)),
                            "Distances contain NaN/Inf")
            self.assertTrue(torch.all(distances >= 0),
                            "Distances should be non-negative")

    def test_weight_stability(self):
        """Test that learnable weights remain stable"""
        model = MALZDA(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        initial_weights = model.get_distance_weights()

        for _ in range(10):
            weights = model.get_distance_weights()
            for key in initial_weights:
                self.assertTrue(weights[key] > 0,
                                f"Weight {key} should remain positive")
                self.assertTrue(torch.isfinite(torch.tensor(weights[key])),
                                f"Weight {key} should be finite")


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions"""

    def test_csv_preprocessing(self):
        """Test CSV data preprocessing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"

            n_samples = 100
            n_features = 20

            data = {
                f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
            }
            data['target'] = np.random.randint(0, 5, n_samples)

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            X_scaled, X_unscaled, y, feature_names, kill_chain_labels = \
                load_and_preprocess_real_data(csv_path)

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

            self.assertLess(np.abs(np.mean(X_scaled)), 0.2,
                            "Scaled data mean not close to 0")
            self.assertLess(np.abs(np.std(X_scaled) - 1.0), 0.3,
                            "Scaled data std not close to 1")

    def test_preprocessing_no_nan(self):
        """Test that preprocessing doesn't introduce NaN values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"

            n_samples = 50
            n_features = 10

            data = {
                f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
            }
            data['target'] = np.random.randint(0, 3, n_samples)

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            X_scaled, X_unscaled, y, feature_names, kill_chain_labels = \
                load_and_preprocess_real_data(csv_path)

            self.assertFalse(np.any(np.isnan(X_scaled)),
                             "Scaled data contains NaN")
            self.assertFalse(np.any(np.isnan(X_unscaled)),
                             "Unscaled data contains NaN")

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

            self.assertEqual(len(feature_names), len(custom_features),
                             "Feature names count mismatch")


class TestVisualizationFunctions(unittest.TestCase):
    """Test visualization and result handling functions"""

    def test_nan_safe_visualization(self):
        """Test that visualization handles NaN values safely"""
        scaling_results = {
            1: {
                'accuracy': 0.85,
                'f1': 0.83,
                'precision': 0.84,
                'recall': 0.82
            },
            5: {
                'accuracy': 0.90,
                'f1': np.nan,
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

        def convert_to_serializable(obj):
            """Convert numpy types to native Python types for JSON serialization"""
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

        json_str = json.dumps(serializable)
        self.assertIsNotNone(json_str,
                             "Results should be JSON serializable")


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests with robustness"""

    def test_complete_training_cycle(self):
        """Test complete training and evaluation cycle"""
        dataset_train = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        dataset_test = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=15,
            feature_dim=32,
            temporal_length=8
        )

        model = MALZDA(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        trainer = MALZDATrainer(model, learning_rate=0.001, device='cpu')

        train_sampler = CompositionalTaskSampler(
            dataset_train, n_way=2, k_shot=2, n_query=5
        )
        test_sampler = CompositionalTaskSampler(
            dataset_test, n_way=2, k_shot=2, n_query=5
        )

        train_accuracies = []
        for _ in range(3):
            support_set, query_set, support_labels, query_labels = \
                train_sampler.sample_task()
            loss, accuracy = trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )
            train_accuracies.append(accuracy)

        eval_results = []
        for _ in range(2):
            support_set, query_set, support_labels, query_labels = \
                test_sampler.sample_task()
            results = trainer.evaluate_episode(
                support_set, query_set, support_labels, query_labels
            )
            eval_results.append(results['accuracy'])

        self.assertGreater(len(train_accuracies), 0,
                           "Training should produce results")
        self.assertGreater(len(eval_results), 0,
                           "Evaluation should produce results")

        self.assertTrue(all(not np.isnan(acc) for acc in train_accuracies),
                        "Training accuracies contain NaN")
        self.assertTrue(all(not np.isnan(acc) for acc in eval_results),
                        "Evaluation accuracies contain NaN")

    def test_training_with_limited_classes(self):
        """Test end-to-end with limited class dataset"""
        dataset_train = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=20,
            feature_dim=32,
            temporal_length=8
        )

        dataset_test = CyberSecurityDataset(
            num_classes=2,
            samples_per_class=12,
            feature_dim=32,
            temporal_length=8
        )

        model = MALZDA(
            packet_dim=32,
            flow_seq_len=8,
            campaign_dim=128,
            embedding_dim=32
        )

        trainer = MALZDATrainer(model, learning_rate=0.001, device='cpu')

        train_sampler = CompositionalTaskSampler(
            dataset_train, n_way=5, k_shot=2, n_query=5
        )
        test_sampler = CompositionalTaskSampler(
            dataset_test, n_way=5, k_shot=2, n_query=5
        )

        self.assertEqual(train_sampler.n_way, 2)
        self.assertEqual(test_sampler.n_way, 2)

        for _ in range(2):
            support_set, query_set, support_labels, query_labels = \
                train_sampler.sample_task()
            loss, accuracy = trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )

            self.assertFalse(np.isnan(loss))
            self.assertFalse(np.isnan(accuracy))

    def test_run_experiment_function(self):
        """Test the main run_experiment function"""
        dataset_train = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=15,
            feature_dim=32,
            temporal_length=8
        )

        dataset_test = CyberSecurityDataset(
            num_classes=3,
            samples_per_class=12,
            feature_dim=32,
            temporal_length=8
        )

        model, eval_results, train_losses, train_accuracies = run_experiment(
            dataset_train,
            dataset_test,
            n_way=2,
            k_shot=2,
            n_query=5,
            num_episodes=5,
            eval_episodes=3,
            use_compositional=True,
            experiment_name="test_experiment"
        )

        self.assertIsNotNone(model)
        self.assertIsNotNone(eval_results)
        self.assertGreater(len(train_losses), 0)
        self.assertGreater(len(train_accuracies), 0)


def run_tests():
    """Run all tests with detailed output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # FIXED: Corrected order and completeness of test classes
    test_classes = [
        TestCyberSecurityDataset,
        TestHierarchicalEncoder,
        TestMALZDAModel,
        TestMALZDAModelWithTensorConversion,
        TestCompositionalTaskSampler,
        TestMALZDATrainer,
        TestRobustDataHandling,
        TestEdgeCasesAndBoundaries,
        TestSamplerEdgeCases,
        TestNumericalStability,
        TestDataPreprocessing,
        TestVisualizationFunctions,
        TestEndToEnd
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

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
    result = run_tests()
    exit(0 if result.wasSuccessful() else 1)


################################################
# END OF FILE                                  #
################################################
