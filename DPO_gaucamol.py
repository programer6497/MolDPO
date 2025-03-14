import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import multiprocessing

# rdkit
from rdkit import Chem, DataStructs

# guacamol
import guacamol
from guacamol.scoring_function import ScoringFunction
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.utils.chemistry import canonicalize_list
from assess_goal_directed_generation import assess_goal_directed_generation

from model import GPT, GPTConfig
from vocabulary import read_vocabulary
from utils import set_seed, sample_SMILES, likelihood, to_tensor, calc_fingerprints

from time import time

def process_group(df):
    # 找到权重最小的行的索引
    idx_min = df['weights'].idxmin()
    # 计算该组中被删去的行数 n = 当前组内行数 - 1
    n = len(df) - 1
    # 将该行的权重减少 n
    df.loc[idx_min, 'weights'] -= n
    if df.loc[idx_min, 'weights']<1:
        df.loc[idx_min, 'weights']=0.1          #防止权重为负或者全部都为0导致choice时异常
    # 返回这一行
    return df.loc[[idx_min]]


class DPO_goal_directed_generator(GoalDirectedGenerator):

    def __init__(self, logger, configs):
        self.load=False
        self.load_memory=False
        self.enable_DPO=False
        
        self.beta=0.1

        self.writer = logger
        self.model_type = configs.model_type
        self.task_id = configs.task_id
        self.num_agents = configs.num_agents
        self.prior_path = configs.prior_path
        self.voc = read_vocabulary(configs.vocab_path)
        self.batch_size = configs.batch_size
        self.n_steps = configs.n_steps
        self.learning_rate = configs.learning_rate
        self.sigma1 = configs.sigma1
        self.sigma2 = configs.sigma2
        # experience replay
        self.memory = pd.DataFrame(columns=["smiles", "scores", "seqs", "fps","weights"])

        if self.load:
            with open("memory.pkl", "rb") as f:
                self.memory = pickle.load(f)
        elif self.load_memory:
            with open("memory.pkl", "rb") as f:
                self.memory = pickle.load(f)
            self.memory["weights"] = 200
        self.memory_size = configs.memory_size
        self.replay = configs.replay
        # penalize similarity
        self.sim_penalize = configs.sim_penalize
        self.sim_thres = configs.sim_thres

        

    def cal_prob(self,seq_y: np.ndarray, model, batch_size=1) -> torch.Tensor:
        input_vector = torch.full((batch_size,1), self.voc["^"],device="cuda",dtype=torch.int32)
        prob = torch.zeros(batch_size, dtype=torch.float32,
                          requires_grad=True,device="cuda")
        done = torch.zeros(batch_size, dtype=torch.int8,device="cuda")
        i = 1
        while not torch.all(done == 1).item() and i <len(seq_y[0]):
            logits, _ = model(input_vector)
            logits = logits[:,-1].squeeze(1)  # 2D
            probabilities = logits.softmax(dim=1)  # 2D
            index = torch.tensor(seq_y[:, i], dtype=torch.int32,device="cuda") 
            new_prob = probabilities[torch.arange(probabilities.size(0)), index]
            new_prob[done == 1] = 1
            prob = prob + torch.log(new_prob)
            done[index == self.voc["$"]] = 1
            input_vector = torch.cat((input_vector, index.unsqueeze(-1)), dim=-1)
            i += 1

        return prob         #实际返回的是log(P)
        

    def _memory_update(self, smiles, scores, seqs,train=True):
        scores = list(scores)
        seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]

        fps_memory = list(self.memory["fps"])

        mean_coef = 0
        for i in range(len(smiles)):
            if scores[i] < 0:
                continue
            # canonicalized SMILES and fingerprints
            fp, smiles_i = calc_fingerprints([smiles[i]])
            new_data = pd.DataFrame({"smiles": smiles_i[0], "scores": scores[i], "seqs": [seqs_list[i]], "fps": fp[0],"weights": 200})
            self.memory = pd.concat([self.memory, new_data], ignore_index=True, sort=False)

            # penalize similarity
            if self.sim_penalize and len(fps_memory) > 0:
                sims = [DataStructs.FingerprintSimilarity(fp[0], x) for x in fps_memory]
                if np.sum(np.array(sims) >= self.sim_thres) > 20:
                	scores[i] = 0

        if train==False:
            self.memory = self.memory.drop_duplicates(subset=["smiles"])
            self.memory = self.memory.sort_values('scores', ascending=False)
            self.memory = self.memory.reset_index(drop=True)
            if len(self.memory) > self.memory_size:
                self.memory = self.memory.head(self.memory_size)
            return


        self.memory = self.memory.groupby('smiles', group_keys=False).apply(process_group)
        self.memory = self.memory.sort_values('scores', ascending=False)
        self.memory = self.memory.reset_index(drop=True)
        if len(self.memory) > self.memory_size:
            self.memory = self.memory.head(self.memory_size)

        # experience replay
        if self.replay > 0:
            s = min(len(self.memory), self.replay)
            experience = self.memory.head(5 * self.replay).sample(s)
            experience = experience.reset_index(drop=True)
            smiles += list(experience["smiles"])
            scores += list(experience["scores"])
            for index in experience.index:
                seqs = torch.cat((seqs, torch.tensor(experience.loc[index, "seqs"], dtype=torch.long).view(1, -1).cuda()), dim=0)

        return smiles, np.array(scores), seqs

    def sample(self,agent,step,ratio=2)
        extra=1 if step<450 else 2
        if (step<100 and step%10==0) or (step<200 and step%5==0) or (step<300 and step%2==0) or step>=300:
            samples, seqs, _ = sample_SMILES(agent, self.voc, n_mols=self.batch_size*10*(ratio-1)*extra)       #采样数据(仅记录在memory中，不会在本轮中被训练)
            scores = scoring_function.score_list(samples)
            self._memory_update(samples, scores, seqs,train=False)            
        return sample_SMILES(agent, self.voc, n_mols=self.batch_size)           #训练数据(记录在memory中，同时在本轮中被训练)


    def generate_optimized_molecules(self, scoring_function, number_molecules, starting_population=None):
        prior_config = GPTConfig(self.voc.__len__(), n_layer=8, n_head=8, n_embd=256, block_size=128)
        prior = GPT(prior_config).to("cuda")
        agents = []
        optimizers = []
        schedulers=[]
        for i in range(self.num_agents):
            agents.append(GPT(prior_config).to("cuda"))
            optimizers.append(agents[i].configure_optimizers(weight_decay=0.1, 
                                                            learning_rate=self.learning_rate, 
                                                            betas=(0.9, 0.95)))
            schedulers.append(CosineAnnealingLR(optimizers[i],T_max=300,eta_min=1e-4))
        if self.load:
            for i in range(self.num_agents):
                agents[i].load_state_dict(torch.load("RL_model_"+str(i)+".pt"), strict=True)  
                optimizers[i].load_state_dict(torch.load("optimizer_"+str(i)+".pth"))
        
        scaler = torch.cuda.amp.GradScaler()
        prior.load_state_dict(torch.load(self.prior_path), strict=True)
        for param in prior.parameters():
            param.requires_grad = False
        prior.eval()
        if not self.load:
            for i in range(self.num_agents):
                agents[i].load_state_dict(torch.load(self.prior_path), strict=True)
                agents[i].eval()

        for step in tqdm(range(self.n_steps)):
            for i in range(self.num_agents):
                samples, seqs, _ = self.sample(agents[i],step)

                scores = scoring_function.score_list(samples)

                samples, scores, seqs = self._memory_update(samples, scores, seqs)

                topKs=[10,50,200,500]
                pos_weights=np.array(self.memory["weights"][0:topKs[i]], dtype=np.float32)       #这里需要指定dtype
                pos_scores=np.array(self.memory["scores"][0:topKs[i]])
                positive_seqs=np.array(self.memory["seqs"][0:topKs[i]])

                if len(pos_weights)<self.batch_size/10:         #刚开始训练时若样本数不足则跳过本轮
                    continue

                # 划分正负样本
                probabilities = pos_weights / np.sum(pos_weights)
                pos_indexes =np.random.choice(len(positive_seqs), size=self.batch_size, replace=True, p=probabilities)
                neg_indexes = np.random.randint(0, len(seqs), self.batch_size)
                pos_batch = np.array([positive_seqs[index] for index in pos_indexes])       
                neg_batch = np.array([seqs[index].cpu() for index in neg_indexes])          #seqs里存的是tensor，需要调用cpu()函数
                pos_scores=torch.tensor([pos_scores[index] for index in pos_indexes], dtype=torch.float32,device="cuda",requires_grad=False)
                neg_scores=torch.tensor([scores[index] for index in neg_indexes], dtype=torch.float32,device="cuda",requires_grad=False)
                prob_w_theta = self.cal_prob(pos_batch, agents[i], self.batch_size)
                prob_l_theta = self.cal_prob(neg_batch, agents[i], self.batch_size)
                with torch.no_grad():
                    prob_w_ref = self.cal_prob(pos_batch, prior, self.batch_size)
                    prob_l_ref = self.cal_prob(neg_batch, prior, self.batch_size)
                factor_function=lambda x:x*1000            #尝试调整，例如factor_function=lambda x: torch.pow(10,(10*(x-0.1)))-0.1 ？
                #TODO：想办法抑制全部趋于最高分，而不探索的情况
                loss = -factor_function((pos_scores-neg_scores))*torch.log(torch.sigmoid(self.beta * (prob_w_theta-prob_w_ref-prob_l_theta+prob_l_ref)))
                loss = loss.mean()
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                schedulers[i].step()

            if step%10==9:
                for i in range(self.num_agents):
                    torch.save(agents[i].state_dict(), "RL_model_"+str(i)+".pt")
                    torch.save(optimizers[i].state_dict(),"optimizer_"+str(i)+".pth")
                with open("memory.pkl", "wb") as f:
                    pickle.dump(self.memory, f)

            if self.task_id in list(range(3, 6)) + list(range(8, 20)):
                self.writer.add_scalar('top-1 score', np.max(np.array(self.memory["scores"])), step)
                self.writer.add_scalar('top-10 score', np.mean(np.array(self.memory["scores"][:10])), step)
                self.writer.add_scalar('top-100 score', np.mean(np.array(self.memory["scores"][:100])), step)
            elif self.task_id in list(range(0, 3)):
                self.writer.add_scalar('top-1 score', np.max(np.array(self.memory["scores"])), step)
            elif self.task_id == 6:
                self.writer.add_scalar('top-159 score', np.mean(np.array(self.memory["scores"][:159])), step)
            elif self.task_id == 7:
                self.writer.add_scalar('top-250 score', np.mean(np.array(self.memory["scores"][:250])), step)

            self.writer.add_scalar('mean score in memory', np.mean(np.array(self.memory["scores"])), step)

        samples_all = canonicalize_list(list(self.memory['smiles']))
        scores_all = scoring_function.score_list(samples_all)
        scored_molecules = zip(samples_all, scores_all)
        assert len(samples_all) == len(scores_all)
        sorted_scored_molecules = sorted(scored_molecules, key=lambda x: (x[1], hash(x[0])), reverse=True)
        top_scored_molecules = sorted_scored_molecules[:number_molecules]

        for i in range(self.num_agents):
            torch.save(agents[i].state_dict(), "RL_model_"+str(i)+".pt")
            torch.save(optimizers[i].state_dict(),"optimizer_"+str(i)+".pth")
        with open("memory.pkl", "wb") as f:
            pickle.dump(self.memory, f)

        return [x[0] for x in top_scored_molecules]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=13)
    parser.add_argument('--num_agents', type=int, default=4)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sigma1', type=float, default=1000)
    parser.add_argument('--sigma2', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--replay', type=int, default=5)
    parser.add_argument('--sim_penalize', type=bool, default=True)
    parser.add_argument('--sim_thres', type=float, default=0.7)
    parser.add_argument('--prior_path', type=str, default="ckpt/guacamol.pt")
    parser.add_argument('--vocab_path', type=str, default="data/vocab.txt")
    parser.add_argument('--output_dir', type=str, default="guacamol_log/")
    args = parser.parse_args()
    print(args)

    set_seed(42)

    writer = SummaryWriter(args.output_dir + f"log_task{args.task_id}/{args.num_agents}_{args.model_type}/")
    if not os.path.exists(args.output_dir + f"results_task{args.task_id}"):
        os.makedirs(args.output_dir + f"results_task{args.task_id}")
    writer.add_text("configs", str(args))

    generator = DPO_goal_directed_generator(logger=writer, configs=args)
    assess_goal_directed_generation(generator, 
        json_output_file=args.output_dir + f"results_task{args.task_id}/{args.num_agents}_{args.model_type}.json", 
        task_id=args.task_id)

    writer.close()
    