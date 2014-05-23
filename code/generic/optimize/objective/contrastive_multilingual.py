__author__ = 'mdenil'

import random
import generic.model.utils


def _parallel_shuffle_lists(*lists):
    pack = zip(*lists)
    random.shuffle(pack)
    return zip(*pack)


class ContrastiveMultilingualEmbeddingObjective(object):
    def __init__(self,
                 tagged_parallel_sequence_provider,
                 n_contrastive_samples,
                 margin,
                 regularizer=None):
        self.tagged_parallel_sequence_provider = tagged_parallel_sequence_provider
        self.n_contrastive_samples = n_contrastive_samples
        self.margin = margin
        self.regularizer = regularizer

    def evaluate(self, tagged_model_collection):
        t1, t2 = random.choice(self.tagged_parallel_sequence_provider.tags)

        m1 = generic.model.utils.ModelEvaluator(
            tagged_model_collection.get_model(t1),
            desired_axes=('b', ('d', 'f', 'w')))
        m2 = generic.model.utils.ModelEvaluator(
            tagged_model_collection.get_model(t2),
            desired_axes=('b', ('d', 'f', 'w')))

        energy_function = self.__class__.Energy()
        loss_function = self.__class__.LossFunction(margin=self.margin)

        p_parallel = self.tagged_parallel_sequence_provider.get_provider((t1, t2))
        x1_clean, meta1_clean, x2_clean, meta2_clean = p_parallel.next_batch()

        total_loss = 0.0
        grads_1_total = [self._zeros_like(p) for p in m1.model.params()]
        grads_2_total = [self._zeros_like(p) for p in m2.model.params()]

        y1_clean = m1.fprop(x1_clean, meta1_clean)
        y2_clean = m2.fprop(x2_clean, meta2_clean)

        e_clean = energy_function.fprop(y1_clean, y2_clean)

        for _ in xrange(self.n_contrastive_samples):
            # new state containers
            m2_noise = generic.model.utils.ModelEvaluator(
                tagged_model_collection.get_model(t2),
                desired_axes=('b', ('d', 'f', 'w')))

            y1_noise = y1_clean

            meta2_noise = dict(meta2_clean)
            x2_noise, meta2_noise['lengths'] = _parallel_shuffle_lists(x2_clean, meta2_clean['lengths'])
            y2_noise = m2_noise.fprop(x2_noise, meta2_noise)

            e_noise = energy_function.fprop(y1_noise, y2_noise)

            total_loss += loss_function.fprop(e_clean, e_noise)

            dloss_clean, dloss_noise = loss_function.bprop(e_clean, e_noise, 1.0)

            denergy_1_clean, denergy_2_clean = energy_function.bprop(y1_clean, y2_clean, dloss_clean)
            denergy_1_noise, denergy_2_noise = energy_function.bprop(y1_noise, y2_noise, dloss_noise)

            grads_1_clean = m1.grads(denergy_1_clean)
            grads_1_noise = m1.grads(denergy_1_noise)

            grads_2_clean = m2.grads(denergy_2_clean)
            grads_2_noise = m2_noise.grads(denergy_2_noise)

            for g, g_clean, g_noise in zip(grads_1_total, grads_1_clean, grads_1_noise):
                g += g_clean
                g += g_noise

            for g, g_clean, g_noise in zip(grads_2_total, grads_2_clean, grads_2_noise):
                g += g_clean
                g += g_noise

        if self.regularizer:
            m = tagged_model_collection.get_model(t1)
            total_loss += self.regularizer.cost(m)
            for g, rg in zip(grads_1_total, self.regularizer.grads(m)):
                g += rg

            m = tagged_model_collection.get_model(t2)
            total_loss += self.regularizer.cost(m)
            for g, rg in zip(grads_2_total, self.regularizer.grads(m)):
                g += rg

        full_grads = tagged_model_collection.full_grads_from_tagged_grads({
            t1: grads_1_total,
            t2: grads_2_total,
        })

        return total_loss, full_grads