"""Optional CPU checks; no pretrained downloads or training datasets required."""
import unittest

try:
    import torch
    from torch_geometric.data import Data
    from ml.gat.model import SiameseGAT, ContrastiveLoss, GATExplainer
    from ml.graph.converter import GraphConverter
    from ml.trajectory.link_predictor import CareerTrajectoryPredictor
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False


@unittest.skipUnless(RESEARCH_AVAILABLE, 'Install optional ML dependencies to run research checks')
class ResearchTests(unittest.TestCase):
    def test_rewriter_checks_do_not_reward_fabrication(self):
        import importlib.util
        if not importlib.util.find_spec('transformers'):
            self.skipTest('Optional transformers dependency is not installed')
        from ml.rewriter.t5_rewriter import factual_checks, T5ResumeRewriter
        self.assertFalse(factual_checks('Built 5 Python services', 'Built 50 Python services')['numbers_preserved'])
        self.assertFalse(factual_checks('Python', 'Python AWS')['recognized_skills_preserved'])
        self.assertFalse(factual_checks('Not deployed to production', 'Deployed to production')['negation_preserved'])
        rewriter = T5ResumeRewriter.__new__(T5ResumeRewriter)
        rewriter.trained = False
        with self.assertRaises(RuntimeError): rewriter.rewrite_bullet('Built Python services')

    def test_legacy_bias_scoring_has_no_fake_probabilities(self):
        import importlib.util
        if not importlib.util.find_spec('transformers'):
            self.skipTest('Optional transformers dependency is not installed')
        from ml.bias.classifier import JDQualityScorer
        result = JDQualityScorer().score_jd('We need a team player with Python experience')
        self.assertNotIn('quality_score', result)
        self.assertNotIn('scores', result)
        self.assertFalse(result['available_learned_model'])
        self.assertFalse(result['bias_detected'])

    def test_negative_pair_has_nonzero_gradient(self):
        resume = torch.tensor([[1.0, 0.0]])
        jd = torch.tensor([[0.8, 0.6]], requires_grad=True)
        criterion = ContrastiveLoss()
        positive = criterion(resume, jd, torch.ones(1))
        negative = criterion(resume, jd, torch.zeros(1))
        self.assertGreater(negative.item(), positive.item())
        negative.backward()
        self.assertGreater(jd.grad.abs().sum().item(), 0)

    def test_attention_is_reproducible_and_from_model(self):
        torch.manual_seed(42)
        model = SiameseGAT({'input_dim': 4, 'hidden_dim': 4, 'output_dim': 4, 'num_heads': 2, 'num_layers': 2})
        graph = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
        explainer = GATExplainer(model)
        first = explainer.explain_match(graph, graph)
        second = explainer.explain_match(graph, graph)
        self.assertTrue(torch.equal(first['attention_weights']['resume'], second['attention_weights']['resume']))
        self.assertTrue(torch.isfinite(first['attention_weights']['resume']).all())

    def test_converter_uses_local_features_and_rejects_empty(self):
        import networkx as nx
        converter = GraphConverter(embedding_dim=64)
        graph = nx.DiGraph()
        graph.add_node('python', type='SKILL', name='Python')
        converted = converter.create_homogeneous_graph(graph)
        self.assertEqual(tuple(converted.x.shape), (1, 64))
        self.assertEqual(tuple(converted.edge_index.shape), (2, 0))
        with self.assertRaises(ValueError):
            converter.create_homogeneous_graph(nx.DiGraph())

    def test_negative_edges_never_include_real_transitions(self):
        predictor = CareerTrajectoryPredictor()
        predictor.career_graph.build_from_career_data([('Junior Engineer', 'Engineer'), ('Engineer', 'Senior Engineer'), ('Senior Engineer', 'Lead')])
        data = predictor.career_graph.get_pyg_data()
        (src, dst), labels = predictor._prepare_training_data(data)
        existing = set(map(tuple, data.edge_index.t().tolist()))
        for source, target, label in zip(src.tolist(), dst.tolist(), labels.tolist()):
            if not label:
                self.assertNotIn((source, target), existing)
                self.assertNotEqual(source, target)


if __name__ == '__main__':
    unittest.main()
