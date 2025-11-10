################################################
'''   Goodnews Daniel (PhD Candidate)
        222166453@student.uj.ac.za
Department of Electrical & Electronics Engineering
Faculty of Engineering & the Built Environment
University of Johannesburg, South Africa        

MAL-ZDA Test Suite
Comprehensive unit tests for all components     '''
################################################

import unittest
import torch
import numpy as np
import tempfile
import json
from pathlib import Path
import pandas as pd

# Import all components from mal_zda
from mal_zda import (
    CyberSecurityDataset,
    HierarchicalEncoder,
    MALZDA,
    CompositionalTaskSampler,
    MALZDATrainer,
    load_and_preprocess_real_data,
    DEVICE
)


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
        expected_len = self.test_config['num_classes'] * self.test_config['samples_per_class']
        self.assertEqual(len(dataset), expected_len)
        
        # Check sample structure
        sample = dataset[0]
        self.assertIn('packet', sample)
        self.assertIn('flow', sample)
        self.assertIn('campaign', sample)
        self.assertIn('class_id', sample)
        self.assertIn('kill_chain', sample)
        
        # Check dimensions
        self.assertEqual(sample['packet'].shape[0], self.test_config['feature_dim'])
        self.assertEqual(sample['flow'].shape, 
                        (self.test_config['temporal_length'], 
                         self.test_config['feature_dim']))
        self.assertEqual(sample['campaign'].shape[0], self.test_config['feature_dim'] * 4)
        
        # Check data types
        self.assertIsInstance(sample['packet'], np.ndarray)
        self.assertIsInstance(sample['flow'], np.ndarray)
        self.assertIsInstance(sample['campaign'], np.ndarray)
        self.assertIsInstance(sample['class_id'], (int, np.integer))
        self.assertIsInstance(sample['kill_chain'], (int, np.integer))
        
        # Check value ranges
        self.assertTrue(0 <= sample['class_id'] < self.test_config['num_classes'])
        self.assertTrue(0 <= sample['kill_chain'] < 5)  # 5 kill-chain phases
    
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
        self.assertEqual(len(dataset), n_samples)
        
        # Check sample structure
        sample = dataset[0]
        self.assertEqual(sample['packet'].shape[0], n_features)
        self.assertEqual(sample['flow'].shape[0], 10)
        self.assertEqual(sample['campaign'].shape[0], n_features * 4)
    
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
        self.assertGreater(len(unique_phases), 1)
        
        # All phases should be valid
        self.assertTrue(all(0 <= p < 5 for p in phases))


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
        self.assertIsNotNone(self.encoder.packet_encoder)
        self.assertIsNotNone(self.encoder.flow_encoder)
        self.assertIsNotNone(self.encoder.campaign_encoder)
        self.assertIsNotNone(self.encoder.layer_norm)
    
    def test_packet_encoder(self):
        """Test packet-level encoder"""
        packet_data = torch.randn(self.batch_size, self.packet_dim)
        packet_emb = self.encoder._encode_packet(packet_data)
        
        self.assertEqual(packet_emb.shape, (self.batch_size, self.embedding_dim))
        self.assertTrue(torch.all(torch.isfinite(packet_emb)))
    
    def test_flow_encoder(self):
        """Test flow-level encoder"""
        flow_data = torch.randn(self.batch_size, self.flow_seq_len, self.packet_dim)
        flow_emb = self.encoder._encode_flow(flow_data)
        
        self.assertEqual(flow_emb.shape, (self.batch_size, self.embedding_dim))
        self.assertTrue(torch.all(torch.isfinite(flow_emb)))
    
    def test_campaign_encoder(self):
        """Test campaign-level encoder"""
        campaign_data = torch.randn(self.batch_size, self.campaign_dim)
        campaign_emb = self.encoder._encode_campaign(campaign_data)
        
        self.assertEqual(campaign_emb.shape, (self.batch_size, self.embedding_dim))
        self.assertTrue(torch.all(torch.isfinite(campaign_emb)))
    
    def test_full_forward_pass(self):
        """Test complete forward pass through all levels"""
        packet_data = torch.randn(self.batch_size, self.packet_dim)
        flow_data = torch.randn(self.batch_size, self.flow_seq_len, self.packet_dim)
        campaign_data = torch.randn(self.batch_size, self.campaign_dim)
        
        packet_emb, flow_emb, campaign_emb = self.encoder(
            packet_data, flow_data, campaign_data
        )
        
        # Check shapes
        self.assertEqual(packet_emb.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(flow_emb.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(campaign_emb.shape, (self.batch_size, self.embedding_dim))
        
        # Check normalization (L2 norm should be ~1)
        packet_norms = torch.norm(packet_emb, p=2, dim=1)
        flow_norms = torch.norm(flow_emb, p=2, dim=1)
        campaign_norms = torch.norm(campaign_emb, p=2, dim=1)
        
        self.assertTrue(torch.allclose(packet_norms, torch.ones_like(packet_norms), atol=1e-5))
        self.assertTrue(torch.allclose(flow_norms, torch.ones_like(flow_norms), atol=1e-5))
        self.assertTrue(torch.allclose(campaign_norms, torch.ones_like(campaign_norms), atol=1e-5))


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
        self.assertIsNotNone(self.model.hierarchical_encoder)
        self.assertIsInstance(self.model.alpha, torch.nn.Parameter)
        self.assertIsInstance(self.model.beta, torch.nn.Parameter)
        self.assertIsInstance(self.model.gamma, torch.nn.Parameter)
        self.assertIsInstance(self.model.temperature, torch.nn.Parameter)
    
    def test_forward_pass(self):
        """Test forward pass through model"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )
        
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        
        support_embeddings, query_embeddings = self.model(support_set, query_set)
        
        # Check structure
        self.assertEqual(len(support_embeddings), 3)
        self.assertEqual(len(query_embeddings), 3)
        
        # Check dimensions
        expected_support_size = self.config['n_way'] * self.config['k_shot']
        self.assertEqual(support_embeddings[0].shape[0], expected_support_size)
        self.assertEqual(support_embeddings[0].shape[1], self.config['embedding_dim'])
    
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
        
        prototypes = self.model.compute_prototypes(support_embeddings, support_labels_tensor)
        
        # Check number of prototypes
        self.assertEqual(len(prototypes), self.config['n_way'])
        
        # Check prototype structure
        for class_id, proto in prototypes.items():
            self.assertIn('packet', proto)
            self.assertIn('flow', proto)
            self.assertIn('campaign', proto)
            self.assertEqual(proto['packet'].shape[0], self.config['embedding_dim'])
    
    def test_distance_computation(self):
        """Test distance computation to prototypes"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query']
        )
        
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        support_embeddings, query_embeddings = self.model(support_set, query_set)
        support_labels_tensor = torch.tensor(support_labels)
        
        prototypes = self.model.compute_prototypes(support_embeddings, support_labels_tensor)
        distances = self.model.compute_distances(query_embeddings, prototypes)
        
        # Check distance matrix shape
        self.assertEqual(distances.shape[0], len(query_set))
        self.assertEqual(distances.shape[1], self.config['n_way'])
        
        # Check distances are non-negative
        self.assertTrue(torch.all(distances >= 0))
    
    def test_distance_weights(self):
        """Test learnable distance weights"""
        weights = self.model.get_distance_weights()
        
        self.assertIn('alpha', weights)
        self.assertIn('beta', weights)
        self.assertIn('gamma', weights)
        self.assertIn('temperature', weights)
        
        # All weights should be positive
        self.assertGreater(weights['alpha'], 0)
        self.assertGreater(weights['beta'], 0)
        self.assertGreater(weights['gamma'], 0)
        self.assertGreater(weights['temperature'], 0)


class TestCompositionalTaskSampler(unittest.TestCase):
    """Test compositional task sampler"""
    
    def setUp(self):
        """Set up sampler configuration"""
        self.dataset = CyberSecurityDataset(
            num_classes=10,
            samples_per_class=50,
            feature_dim=64,
            temporal_length=10
        )
        
        self.config = {
            'n_way': 5,
            'k_shot': 2,
            'n_query': 10
        }
    
    def test_compositional_sampling(self):
        """Test compositional task sampling with kill-chain awareness"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=True
        )
        
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        
        # Check support set size
        expected_support_size = self.config['n_way'] * self.config['k_shot']
        self.assertEqual(len(support_set), expected_support_size)
        self.assertEqual(len(support_labels), expected_support_size)
        
        # Check number of unique classes
        unique_support_classes = len(set(support_labels))
        self.assertEqual(unique_support_classes, self.config['n_way'])
        
        # Check kill-chain phase diversity in support set
        support_phases = [sample['kill_chain'] for sample in support_set]
        unique_phases = len(set(support_phases))
        self.assertGreater(unique_phases, 1, "Should have phase diversity")
    
    def test_standard_sampling(self):
        """Test standard sampling without kill-chain awareness"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=False
        )
        
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        
        # Check basic structure
        self.assertEqual(len(support_set), self.config['n_way'] * self.config['k_shot'])
        self.assertEqual(len(set(support_labels)), self.config['n_way'])
    
    def test_query_set_validity(self):
        """Test query set does not overlap with support set"""
        sampler = CompositionalTaskSampler(
            self.dataset,
            n_way=self.config['n_way'],
            k_shot=self.config['k_shot'],
            n_query=self.config['n_query'],
            include_kill_chain=True
        )
        
        support_set, query_set, support_labels, query_labels = sampler.sample_task()
        
        # Get indices (if available)
        support_ids = [id(s) for s in support_set]
        query_ids = [id(q) for q in query_set]
        
        # Check no overlap
        overlap = set(support_ids) & set(query_ids)
        self.assertEqual(len(overlap), 0, "Support and query sets should not overlap")


class TestMALZDATrainer(unittest.TestCase):
    """Test training and evaluation"""
    
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
        
        self.trainer = MALZDATrainer(self.model, learning_rate=0.001, device='cpu')
        
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
        self.assertIsInstance(loss, float)
        self.assertIsInstance(accuracy, float)
        
        # Check value ranges
        self.assertGreater(loss, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_evaluation_episode(self):
        """Test single evaluation episode"""
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        
        results = self.trainer.evaluate_episode(
            support_set, query_set, support_labels, query_labels
        )
        
        # Check result structure
        self.assertIn('loss', results)
        self.assertIn('accuracy', results)
        self.assertIn('f1', results)
        self.assertIn('precision', results)
        self.assertIn('recall', results)
        self.assertIn('predictions', results)
        self.assertIn('labels', results)
        self.assertIn('distance_weights', results)
        
        # Check metric ranges
        self.assertGreaterEqual(results['accuracy'], 0)
        self.assertLessEqual(results['accuracy'], 1)
        self.assertGreaterEqual(results['f1'], 0)
        self.assertLessEqual(results['f1'], 1)
    
    def test_training_history(self):
        """Test training history recording"""
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        
        # Train for a few episodes
        for _ in range(3):
            self.trainer.train_episode(
                support_set, query_set, support_labels, query_labels
            )
        
        # Check history
        self.assertEqual(len(self.trainer.training_history['loss']), 3)
        self.assertEqual(len(self.trainer.training_history['accuracy']), 3)
        self.assertEqual(len(self.trainer.training_history['distance_weights']), 3)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        # Train for a few steps
        support_set, query_set, support_labels, query_labels = self.sampler.sample_task()
        self.trainer.train_episode(support_set, query_set, support_labels, query_labels)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model.pt"
            self.trainer.save_model(save_path)
            
            # Check file exists
            self.assertTrue(save_path.exists())
            
            # Load model
            new_model = MALZDA(
                packet_dim=self.config['packet_dim'],
                flow_seq_len=self.config['flow_seq_len'],
                campaign_dim=self.config['campaign_dim'],
                embedding_dim=self.config['embedding_dim']
            )
            new_trainer = MALZDATrainer(new_model, learning_rate=0.001, device='cpu')
            new_trainer.load_model(save_path)
            
            # Check history was loaded
            self.assertGreater(len(new_trainer.training_history['loss']), 0)


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
            self.assertEqual(X_scaled.shape[0], n_samples)
            self.assertEqual(X_scaled.shape[1], n_features)
            self.assertEqual(len(y), n_samples)
            self.assertEqual(len(feature_names), n_features)
            self.assertEqual(len(kill_chain_labels), n_samples)
            
            # Check scaling (mean ~0, std ~1)
            self.assertTrue(np.abs(np.mean(X_scaled)) < 0.1)
            self.assertTrue(np.abs(np.std(X_scaled) - 1.0) < 0.2)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_complete_training_cycle(self):
        """Test complete training and evaluation cycle"""
        # Create dataset
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
        self.assertGreater(len(train_accuracies), 0)
        self.assertGreater(len(eval_results), 0)
        
        # Check reasonable performance (random baseline ~33% for 3-way)
        mean_train_acc = np.mean(train_accuracies)
        mean_eval_acc = np.mean(eval_results)
        
        self.assertGreater(mean_train_acc, 0.2, "Training accuracy too low")
        self.assertGreater(mean_eval_acc, 0.2, "Evaluation accuracy too low")


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCyberSecurityDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestHierarchicalEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestMALZDAModel))
    suite.addTests(loader.loadTestsFromTestCase(TestCompositionalTaskSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestMALZDATrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result


if __name__ == '__main__':
    # Run tests
    result = run_tests()
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)


################################################
# END OF FILE                                  #
################################################