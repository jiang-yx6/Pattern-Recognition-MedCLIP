"""Mathematical and data-boundary checks; contains no real MIMIC data."""
import unittest
import numpy as np
import torch
from mimic_pipeline.common import seed_all
from mimic_pipeline.metric import MetricHead, auxiliary_losses, contrastive_loss, negative_weights, ranks, transform_cache, valid_negatives
from mimic_pipeline.geometry import space_statistics


class PipelineTests(unittest.TestCase):
    def test_infonce_matches_analytic_two_item_case(self):
        v=torch.eye(2,requires_grad=True)
        labels=torch.zeros((2,5),dtype=torch.long);mask=torch.zeros_like(labels);ids=torch.arange(2)
        loss,skipped,_=contrastive_loss(v,v,labels,mask,ids,temperature=1.)
        self.assertAlmostEqual(loss.item(),float(np.log1p(np.exp(-1))),places=6)
        self.assertEqual(skipped,0);loss.backward();self.assertTrue(torch.isfinite(v.grad).all())
        hard,_,_=contrastive_loss(v,v,labels,mask,ids,temperature=1.,hard=True)
        self.assertAlmostEqual(hard.item(),float(np.log1p(2*np.exp(-1))),places=6)

    def test_unknown_labels_are_not_treated_as_identical_positive_sets(self):
        labels=torch.tensor([[1,0],[1,0],[0,0],[0,0]])
        masks=torch.tensor([[1,1],[1,1],[0,0],[0,0]])
        valid=valid_negatives(labels,masks,torch.arange(4))
        self.assertFalse(valid[0,1]);self.assertTrue(valid[2,3]);self.assertFalse(valid.diagonal().any())

    def test_no_valid_negatives_is_explicitly_skipped(self):
        loss,skipped,_=contrastive_loss(torch.eye(2),torch.eye(2),torch.ones((2,1)),torch.ones((2,1)),torch.arange(2))
        self.assertIsNone(loss);self.assertEqual(skipped,4)

    def test_hard_budget_uses_only_valid_negatives(self):
        scores=torch.tensor([[999.,.1,.9,.3],[.5,999.,.2,.3],[99.,99.,99.,99.],[.1,.4,.6,999.]])
        valid=torch.tensor([[False,True,True,True],[True,False,False,False],[False,False,False,False],[True,True,True,False]])
        weights,hard=negative_weights(scores,valid,.25,2.)
        self.assertEqual(hard.sum(1).tolist(),[1,1,0,1]);self.assertTrue(hard[0,2])
        self.assertTrue(torch.equal(weights.diagonal(),torch.ones(4)))

    def test_fixed_order_breaks_retrieval_ties(self):
        np.testing.assert_array_equal(ranks(np.zeros((3,3))),[1,2,3])
        np.testing.assert_array_equal(ranks(np.eye(3)),[1,1,1])
        with self.assertRaises(ValueError):ranks(np.ones((2,3)))

    def test_prompt_ensemble_applies_head_before_normalize_and_mean(self):
        head=MetricHead(2,2,'A')
        with torch.no_grad():
            head.image_shared.weight.copy_(torch.eye(2));head.image_shared.bias.zero_()
            head.text_shared.weight.copy_(torch.tensor([[2.,0.],[0.,1.]]));head.text_shared.bias.copy_(torch.tensor([.2,.4]))
        variants=np.array([[[1.,0.]],[[0.,1.]],[[1.,0.]],[[0.,1.]],[[.6,.8]]],dtype='float32')
        cache={'image_feature':np.array([[1.,0.]],dtype='float32'),'text_prompt_features':variants}
        _,actual=transform_cache(cache,head,torch.device('cpu'),5)
        with torch.no_grad():
            expected=torch.nn.functional.normalize(torch.nn.functional.normalize(head.text_shared(torch.from_numpy(variants)),dim=-1).mean(0),dim=-1).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-7)

    def test_shared_private_and_reconstruction_all_have_gradients(self):
        head=MetricHead(4,3,'C');v=torch.nn.functional.normalize(torch.randn(6,4),dim=-1);t=torch.nn.functional.normalize(torch.randn(6,4),dim=-1)
        encoded=head(v,t);reconstruction,covariance=auxiliary_losses(encoded,v,t)
        contrast,_,_=contrastive_loss(encoded['image_feature'],encoded['text_feature'],torch.zeros((6,5)),torch.zeros((6,5)),torch.arange(6))
        (contrast+.1*reconstruction+.01*covariance).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in head.parameters()))

    def test_geometry_on_identical_orthogonal_modalities(self):
        x=np.eye(4,dtype='float32');labels=np.zeros((4,5),dtype=int);labels[:2,0]=1;labels[2:,1]=1
        metrics,_=space_statistics(x,x,labels,np.ones_like(labels))
        self.assertEqual(metrics['alignment_squared_l2'],0.);self.assertEqual(metrics['modality_gap'],0.)
        self.assertAlmostEqual(metrics['image_uniformity'],-4.,places=6)
        self.assertEqual(metrics['image_dimensions_90'],3)

    def test_seed_repeats_initialization(self):
        seed_all(42);a=MetricHead(4,3,'D')
        seed_all(42);b=MetricHead(4,3,'D')
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))


if __name__=='__main__':
    unittest.main()
