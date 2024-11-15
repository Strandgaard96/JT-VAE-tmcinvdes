import copy
import math
from collections import defaultdict

import rdkit.Chem as Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import DataStructs
from rdkit.Chem import AllChem

from .chemutils import (
    attach_mols,
    copy_edit_mol,
    enum_assemble,
    is_valid_smiles,
    set_atommap,
)
from .datautils_prop import get_tensors
from .jtmpn import JTMPN
from .jtnn_dec import JTNNDecoder
from .jtnn_enc import JTNNEncoder
from .mol_tree import MolTree
from .mpn import MPN
from .nnutils import create_var


class JTpropVAE(nn.Module):
    def __init__(
        self,
        vocab,
        hidden_size,
        latent_size,
        depthT,
        depthG,
        train_mode,
    ):
        super(JTpropVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size = (
            latent_size // 2
        )  # Tree and Mol has two vectors

        self.jtnn = JTNNEncoder(
            hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size)
        )
        self.decoder = JTNNDecoder(
            vocab, hidden_size, latent_size, nn.Embedding(vocab.size(), hidden_size)
        )

        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        self.A_assm = nn.Linear(latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(hidden_size, latent_size)
        self.T_var = nn.Linear(hidden_size, latent_size)
        self.G_mean = nn.Linear(hidden_size, latent_size)
        self.G_var = nn.Linear(hidden_size, latent_size)

        self.propNN = nn.Sequential(
            nn.Linear(self.latent_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 2),
        )

        self.train_mode = train_mode

        for term in self.train_mode:
            if term == "denticity":
                # Binary classification (0 for monodentate and 1 for bidentate)
                self.denticityNN = nn.Sequential(
                    nn.Linear(self.latent_size * 2, self.hidden_size),
                    nn.Sigmoid(),
                    nn.Linear(self.hidden_size, 1),
                    # The outptut is >0.5 for bidentate and <0.5 for bidentate
                )
                self.denticityNN_loss = nn.BCEWithLogitsLoss()
                print("Added denticity layer")
            elif term == "isomer":
                # Binary classification (0 for cis and 1 for trans)
                self.isomerNN = nn.Sequential(
                    nn.Linear(self.latent_size * 2, self.hidden_size),
                    nn.Sigmoid(),
                    nn.Linear(self.hidden_size, 3),  # Number of output classes
                )
                self.isomerNN_loss = nn.CrossEntropyLoss()
                # self.isomerNN_loss = nn.BCEWithLogitsLoss()
                print("Added isomer layer")
            else:
                raise ValueError("train_mode is not a valid value")

        self.prop_loss = nn.MSELoss()

        # Utility class for analyzing optimizations
        self.analyzer = Analyzer(self)

    def param_norm(self):
        return math.sqrt(sum([(p.norm().item() ** 2) for p in self.parameters()]))

    def grad_norm(self):
        return math.sqrt(
            sum(
                [
                    (p.grad.norm().item() ** 2)
                    for p in self.parameters()
                    if p.grad is not None
                ]
            )
        )

    def encode(self, jtenc_holder, mpn_holder):
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    # def encode_latent_from_smiles(self, smiles_list, props):
    #     tree_batch = []
    #     for i, elem in enumerate(smiles_list):
    #         s = MolTree(elem)
    #         tree_batch.append(s)
    #     _, _, jtenc_holder, mpn_holder_ = get_tensors(
    #         tree_batch, props, self.vocab, assm=False
    #     )
    #     z1, z2 = self.encode_latent(jtenc_holder, mpn_holder)
    #     return z1, z2

    def encode_latent(self, jtenc_holder, mpn_holder):
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat(
            [tree_var, mol_var], dim=1
        )

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.
        kl_loss = (
            -0.5
            * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var))
            / batch_size
        )
        epsilon = create_var(torch.randn_like(z_mean))
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def sample_prior(self, prob_decode=False):
        z_tree = torch.randn(1, self.latent_size).cuda()
        z_mol = torch.randn(1, self.latent_size).cuda()
        return self.decode(z_tree, z_mol, prob_decode)

    def optimize_denticity(
        self,
        x_batch,
        sim_cutoff,
        lr=0.2,
        num_iter=20,
        prob_decode=False,
        desired_denticity="monodentate",
    ):
        """Optimize a given ligand in latent space.

        Parameters
        ----------
        x_batch : list
            List containing the pre-processed data for a ligand
        sim_cutoff : float
            Tanimoto similarity cutoff for the optimization.
        lr : float
            Learning rate for the optimization.
        num_iter : int, optional
            Number of steps along the gradient in latent space.
        prob_decode : bool
            Decoding param. Unclear what it does. Keep False
        desired_denticity : str, optional
            Which denticity to optimize towards

        Returns
        -------
        new_smiles : str
            The SMILES string of the optimized molecule.
        tanimoto : float
            The Tanimoto similarity between the optimized molecule and the starting molecule.
        """
        x_batch, x_prop, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)

        mol = Chem.MolFromSmiles(x_batch[0].smiles)
        AllChem.GetMorganFingerprint(mol, 2)

        z_tree_mean = self.T_mean(x_tree_vecs)
        z_mol_mean = self.G_mean(x_mol_vecs)

        mean = torch.cat([z_tree_mean, z_mol_mean], dim=1)
        cur_vec = create_var(mean.data, True)

        visited = []
        cuda0 = torch.device("cuda:0")

        # Apply different scaling on the gradient along the homo-lumo gap direction.
        1.5 * lr

        # Predictions holder
        predictions = defaultdict(list)
        tree_vec, mol_vec = torch.chunk(cur_vec, 2, dim=1)
        new_smiles = self.decode(tree_vec, mol_vec, prob_decode=prob_decode)
        print(x_batch[0].smiles, "encoded smiles: ", new_smiles)

        # Step in gradient space
        sig = nn.Sigmoid()
        for step in range(num_iter):
            # Now we get the gradients of the denticity layer.
            denticity_gradient = torch.tensor([], dtype=torch.float32, device=cuda0)
            denticity_prediction = self.denticityNN(cur_vec)
            predictions["denticity"].append(denticity_prediction)

            denticity_gradient = torch.autograd.grad(
                denticity_prediction, cur_vec, retain_graph=True
            )[0]
            denticity_gradient = denticity_gradient.squeeze()
            # Normalize the denticity gradient
            denticity_norm = torch.nn.functional.normalize(denticity_gradient, dim=-1)
            # Normalize the denticity gradient

            # Add denticity gradient in given direction
            if desired_denticity == "monodentate":  # Decrease probability
                cur_vec = cur_vec - lr * denticity_norm * 2
            elif desired_denticity == "bidentate":  # Increase probability
                cur_vec = cur_vec + lr * denticity_norm * 2
            else:
                raise ValueError
            #
            cur_vec = create_var(cur_vec, True)
            visited.append(cur_vec)

        # Now we try to get the first latent vector that has a bidentate prediction
        for i, elem in enumerate(visited):
            if sig(self.denticityNN(elem)) > 0.5:
                print(f"Found new denticity at {i}")
                break

        tree_vec, mol_vec = torch.chunk(elem, 2, dim=1)
        new_smiles = self.decode(tree_vec, mol_vec, prob_decode=prob_decode)

        # Check the SMILES for encoding atoms
        if not is_valid_smiles(new_smiles):
            print("Not valid smiles after denticity optimization")
        return new_smiles

    def optimize(
        self,
        x_batch,
        sim_cutoff,
        lr=0.2,
        num_iter=20,
        type="charge",
        prob_decode=False,
        minimize=True,
    ):
        """Optimize a given ligand in latent space.

        Parameters
        ----------
        x_batch : list
            List containing the pre-processed data for a ligand
        sim_cutoff : float
            Tanimoto similarity cutoff for the optimization.
        lr : float
            Learning rate for the optimization.
        num_iter : int, optional
            Number of steps along the gradient in latent space.
        type : str
            The property to optimize
        prob_decode : bool
            Decoding param. Unclear what it does. Keep False
        minimize : bool
            Whether to minimize or maximize the property.

        Returns
        -------
        new_smiles : str
            The SMILES string of the optimized molecule.
        tanimoto : float
            The Tanimoto similarity between the optimized molecule and the starting molecule.
        """
        x_batch, x_prop, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)

        mol = Chem.MolFromSmiles(x_batch[0].smiles)
        start_fingerprint = AllChem.GetMorganFingerprint(mol, 2)

        z_tree_mean = self.T_mean(x_tree_vecs)
        z_mol_mean = self.G_mean(x_mol_vecs)

        mean = torch.cat([z_tree_mean, z_mol_mean], dim=1)
        cur_vec = create_var(mean.data, True)

        visited = []
        cuda0 = torch.device("cuda:0")

        # Apply different scaling on the gradient along the homo-lumo gap direction.
        lr_homo = 1.5 * lr

        # Predictions holder
        predictions = defaultdict(list)

        # Step in gradient space
        for step in range(num_iter):
            # Get property prediction
            prop_val = self.propNN(cur_vec)
            predictions["homo-lumo-gap"].append(prop_val.tolist()[0][0])
            predictions["Ir-cm5"].append(prop_val.tolist()[0][1])

            # NB THE TWO LINES BELOW IS A MORE MODERN WAY OF GETTING GRADIENT. HAVE NOT TESTED BUT IT REMOVES THE NEED FOR A FOR LOOP AND
            # IS MORE TRANSPARENT. dydx3 is equal to [grad,grad2]
            # grad = torch.autograd.grad(prop_val, cur_vec,grad_outputs=torch.tensor([[1,0]],device=torch.device('cuda:0')),retain_graph=True) first property gradient
            # grad2 = torch.autograd.grad(prop_val, cur_vec,grad_outputs=torch.tensor([[0,1]],device=torch.device('cuda:0')),retain_graph=True) second property gradient

            gradient_holder = torch.tensor([], dtype=torch.float32, device=cuda0)
            # For each property in prop_val prediction, get gradient.
            for i in range(2):
                li = torch.zeros_like(prop_val)
                li[:, i] = 1.0
                d = torch.autograd.grad(
                    prop_val, cur_vec, retain_graph=True, grad_outputs=li
                )[
                    0
                ]  # dydx: (batch_size, input_dim)
                gradient_holder = torch.concat(
                    (gradient_holder, d.unsqueeze(dim=1)), dim=1
                )

            gradient_holder = gradient_holder.squeeze()

            # Now we get the gradients of the denticity layer.
            # The scaler decides the gradient direction
            scaler = -1 if minimize else 1

            # Normalize the q,epsilon gradients
            norm0 = torch.nn.functional.normalize(gradient_holder[0], dim=-1)
            norm1 = torch.nn.functional.normalize(gradient_holder[1], dim=-1)

            # Normalize the denticity gradient
            # Normalize the denticity gradient
            # mono_norm = torch.nn.functional.normalize(dent_dy[0], dim=-1)
            # bi_norm = torch.nn.functional.normalize(dent_dy[1], dim=-1)

            # Get the gradient magnitudes
            # n0 = torch.linalg.vector_norm(dydx3[0])
            # n1 = torch.linalg.vector_norm(dydx3[1])

            # Add a gradient step in a given direction
            if type == "both":
                cur_vec = cur_vec.data + scaler * lr * norm1 + scaler * lr_homo * norm0
            elif type == "homo_lumo_gap":
                cur_vec = cur_vec.data + scaler * lr_homo * norm0
            elif type == "charge":
                cur_vec = cur_vec.data + scaler * lr * norm1
            elif type == "both_opposite":
                cur_vec = cur_vec.data + scaler * lr_homo * norm0 - scaler * lr * norm1
            else:
                raise ValueError

            # Add denticity gradient in given direction
            cur_vec = create_var(cur_vec, True)
            visited.append(cur_vec)

        # Now we want to get the best possible latent vectors.
        steps_along_gradient = []
        li, r = 0, num_iter - 1
        counter = 1
        while li < r - 1:
            mid = (li + r) // 2
            new_vec = visited[mid]
            tree_vec, mol_vec = torch.chunk(new_vec, 2, dim=1)
            new_smiles = self.decode(tree_vec, mol_vec, prob_decode=prob_decode)

            if new_smiles is None:
                r = mid - 1
                continue

            new_mol = Chem.MolFromSmiles(new_smiles)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(start_fingerprint, fp2)

            # Store all the relevant data in one place
            steps_along_gradient.append(
                {
                    "gradient_step": counter,
                    "current_similarity": sim,
                    "latent_vector_idx": mid,
                    "current_smiles": new_smiles,
                }
            )

            # If under cutoff we move backwards
            if sim < sim_cutoff:
                r = mid - 1
            else:
                li = mid
            counter += 1

        # Check the SMILES for encoding atoms
        if not is_valid_smiles(new_smiles):
            # Sort the results list in place by latent vector indec in descending order. We start from furthest away
            # and then move backwards to find candidates satisfying the tanimoto cutoff.
            steps_along_gradient.sort(
                key=lambda x: x["latent_vector_idx"], reverse=True
            )

            # If not we select on of the others that satisfied the tanimoto criteria.
            for res in steps_along_gradient:
                if res["current_similarity"] > sim_cutoff:
                    candidate = res["current_smiles"]
                    # Check if valid
                    if is_valid_smiles(candidate):
                        new_smiles = candidate
                        break
                    else:
                        print(
                            "None of the tanimoto candidates passed the validity check"
                        )
                        new_smiles = None

        # self.analyzer.print_smiles_gradient()

        # Collect run results:
        run_results = {}
        run_results["new_smiles"] = new_smiles
        run_results["steps_along_gradient"] = steps_along_gradient
        run_results["predictions"] = predictions
        run_results["latent_vectors"] = [x.tolist() for x in visited]
        run_results["tanimoto_similarity"] = None

        # Return the original smiles if we did not find an optimized smiles
        if new_smiles is None:
            return run_results

        # Get similarity of chosen smiles
        new_mol = Chem.MolFromSmiles(new_smiles)
        fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
        sim = DataStructs.TanimotoSimilarity(start_fingerprint, fp2)
        run_results["tanimoto_similarity"] = sim
        if sim >= sim_cutoff:
            return run_results
        else:
            run_results["new_smiles"] = None
            print("Molecule could not satisfy tanimoto cutoff")
            return run_results

    def forward(self, x_batch, beta):
        x_batch, x_prop_holder, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch

        length_p = len(x_prop_holder[0])
        # The first two entries are the q,epsilon properties
        x_prop = x_prop_holder[:, 0:2]

        # self.analyzer.print_smiles_gradient("test")

        # If we only have two properties, dont attempt to unpack the others
        if length_p == 2:
            pass
        elif length_p == 3:
            # The third value should be denticity
            x_dent = x_prop_holder[:, 2]
        elif length_p == 4:
            # The third value should be denticity
            x_dent = x_prop_holder[:, 2]
            # The fourth value should be isomer
            x_isomer = x_prop_holder[:, 3]
        else:
            raise ValueError("The correct number of properties could not be unpacked")

        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)

        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch, z_tree_vecs)
        assm_loss, assm_acc = self.assm(
            x_batch, x_jtmpn_holder, z_mol_vecs, x_tree_mess
        )

        # Learn properties
        all_vec = torch.cat([z_tree_vecs, z_mol_vecs], dim=1)
        prop_label = create_var(x_prop[:, 0:2])
        prop_loss = self.prop_loss(self.propNN(all_vec).squeeze(), prop_label)

        for term in self.train_mode:
            if term == "denticity":
                # Learn dendicity
                x_dent = x_dent.type(
                    torch.FloatTensor
                )  # This is done in order to have it work with the cross entropy loss. Otherwise there is an error.
                denticity_label = create_var(x_dent)
                denticity_label.int()
                dent_loss = self.denticityNN_loss(
                    self.denticityNN(all_vec).squeeze(), denticity_label
                )
            elif term == "isomer":
                # Learn isomer
                x_isomer = x_isomer.type(
                    torch.FloatTensor
                )  # This is done in order to have it work with the cross entropy loss. Otherwise there is an error.
                isomer_label = create_var(x_isomer)
                isomer_label.int()
                isomer_loss = self.isomerNN_loss(
                    self.isomerNN(all_vec).squeeze(), isomer_label
                )

        # Standard loss
        standard_loss = word_loss + topo_loss + assm_loss + beta * kl_div + prop_loss
        # Standard log
        log_metrics = {
            "kl_div": kl_div.item(),
            "word_acc": word_acc * 100,
            "topo_acc": topo_acc * 100,
            "assm_acc": assm_acc * 100,
            "prop_loss": prop_loss.item() * 100,
        }

        # Add terms to logging and loss
        for term in self.train_mode:
            if term == "denticity":
                standard_loss += dent_loss
                log_metrics["dent_loss"] = dent_loss.item() * 100
            elif term == "isomer":
                standard_loss += isomer_loss
                log_metrics["isomer_loss"] = isomer_loss.item() * 100

        return standard_loss, log_metrics

    def assm(self, mol_batch, jtmpn_holder, x_mol_vecs, x_tree_mess):
        jtmpn_holder, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
        batch_idx = create_var(batch_idx)

        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)

        x_mol_vecs = x_mol_vecs.index_select(0, batch_idx)
        x_mol_vecs = self.A_assm(x_mol_vecs)  # bilinear
        scores = torch.bmm(x_mol_vecs.unsqueeze(1), cand_vecs.unsqueeze(-1)).squeeze()

        cnt, tot, acc = 0, 0, 0
        all_loss = []
        for i, mol_tree in enumerate(mol_batch):
            comp_nodes = [
                node
                for node in mol_tree.nodes
                if len(node.cands) > 1 and not node.is_leaf
            ]
            cnt += len(comp_nodes)
            for node in comp_nodes:
                label = node.cands.index(node.label)
                ncand = len(node.cands)
                cur_score = scores.narrow(0, tot, ncand)
                tot += ncand

                if cur_score.data[label] >= cur_score.max().item():
                    acc += 1

                label = create_var(torch.LongTensor([label]))
                all_loss.append(self.assm_loss(cur_score.view(1, -1), label))

        all_loss = sum(all_loss) / len(mol_batch)
        return all_loss, acc * 1.0 / cnt

    def decode(self, x_tree_vecs, x_mol_vecs, prob_decode):
        # currently do not support batch decoding
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        pred_root, pred_nodes = self.decoder.decode(x_tree_vecs, prob_decode)
        if len(pred_nodes) == 0:
            return None
        elif len(pred_nodes) == 1:
            return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = len(node.neighbors) == 1
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (
            tree_mess,
            mess_dict,
        )  # Important: tree_mess is a matrix, mess_dict is a python dict

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze()  # bilinear

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for node in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol, _ = self.dfs_assemble(
            tree_mess,
            x_mol_vecs,
            pred_nodes,
            cur_mol,
            global_amap,
            [],
            pred_root,
            None,
            prob_decode,
            check_aroma=True,
        )
        if cur_mol is None:
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for node in pred_nodes]
            global_amap[1] = {
                atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()
            }
            cur_mol, pre_mol = self.dfs_assemble(
                tree_mess,
                x_mol_vecs,
                pred_nodes,
                cur_mol,
                global_amap,
                [],
                pred_root,
                None,
                prob_decode,
                check_aroma=False,
            )
            if cur_mol is None:
                cur_mol = pre_mol

        if cur_mol is None:
            return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None

    def dfs_assemble(
        self,
        y_tree_mess,
        x_mol_vecs,
        all_nodes,
        cur_mol,
        global_amap,
        fa_amap,
        cur_node,
        fa_node,
        prob_decode,
        check_aroma,
    ):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = list(zip(*cands))
        aroma_score = torch.Tensor(aroma_score).cuda()
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(
                fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0]
            )
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_score
        else:
            scores = torch.Tensor([1.0])

        if prob_decode:
            try:
                probs = (
                    F.softmax(scores.view(1, -1), dim=1).squeeze() + 1e-7
                )  # prevent prob = 0
                cand_idx = torch.multinomial(probs, probs.numel())
            except Exception:
                _, cand_idx = torch.sort(scores, descending=True)
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid:
                    continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][
                    ctr_atom
                ]

            cur_mol = attach_mols(
                cur_mol, children, [], new_global_amap
            )  # father is already attached
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None:
                continue

            has_error = False
            for nei_node in children:
                if nei_node.is_leaf:
                    continue
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess,
                    x_mol_vecs,
                    all_nodes,
                    cur_mol,
                    new_global_amap,
                    pred_amap,
                    nei_node,
                    cur_node,
                    prob_decode,
                    check_aroma,
                )
                if tmp_mol is None:
                    has_error = True
                    if i == 0:
                        pre_mol = tmp_mol2
                    break
                cur_mol = tmp_mol

            if not has_error:
                return cur_mol, cur_mol

        return None, pre_mol


class Analyzer:
    def __init__(self, model):
        self.jt_vae = model
        return

    def print_smiles_gradient(
        self, visited_vectors, selected_idx=None, prob_decode=None, fp1=None
    ):
        # Print all the smiles along the gradient.
        to_print = []
        for v in visited_vectors[0:selected_idx]:
            tree_vec, mol_vec = torch.chunk(v, 2, dim=1)
            new_smiles = self.jt_vae.decode(tree_vec, mol_vec, prob_decode=prob_decode)

            prop_val = self.jt_vae.propNN(v)
            # print(prop_val)
            prediction = prop_val.tolist()

            new_mol = Chem.MolFromSmiles(new_smiles)
            # Chem.AssignStereochemistry(new_mol)
            fp2 = AllChem.GetMorganFingerprint(new_mol, 2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
            # print(sim,x_batch[0].smiles,new_smiles)

            to_print.append((new_smiles, prediction, sim))
        print(to_print)
        return
