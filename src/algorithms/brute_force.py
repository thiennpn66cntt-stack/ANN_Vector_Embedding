
import numpy as np
import torch


def exact_search(query_vecs, database_vecs, k=10, batch_size=32):

    db_t = torch.from_numpy(database_vecs)

    if torch.cuda.is_available():
        db_t = db_t.to('cuda')

    all_ids = []
    all_scores = []

    for start in range(0, len(query_vecs), batch_size):

        q_t = torch.from_numpy(
            query_vecs[start:start+batch_size]
        )

        if torch.cuda.is_available():
            q_t = q_t.to('cuda')

        scores = q_t @ db_t.T

        vals, ids = torch.topk(
            scores,
            k=k,
            dim=1,
            largest=True,
            sorted=True
        )

        all_ids.append(ids.cpu().numpy())
        all_scores.append(vals.cpu().numpy())

    return np.vstack(all_ids), np.vstack(all_scores)
