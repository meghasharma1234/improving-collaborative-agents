from agent_embedding.train_trajRNN import TrainerTrajRNN   
# from self_play_baseline.train_overcooked_sp import trainSelfPlay
from AIRL.airl import AIRLTrainer

from pathlib import Path
import yaml

from collections import Counter


if __name__ == '__main__':
    hparams = yaml.safe_load(Path('./hparams/hparams.yaml').read_text())

    ####################################################    
    ### 1. RNN Agent embedding trainning
    ####################################################    
    # trainer = TrainerTrajRNN(hparams)
    # trainer.train()

    ####################################################
    ### 2. train self play agent
    ####################################################
    # airl_trainer = AIRLTrainer(hparams=hparams)
    # expert = airl_trainer.trainExpertPolicy()
    # airl_trainer.evaluatePolicy(policy=expert)


    ####################################################
    ### 3. gen. rollout & train AIRL & evalute learned policy
    ####################################################
    airl_trainer = AIRLTrainer(hparams=hparams)
    
    ### get rollouts
    # expert = airl_trainer.trainExpertPolicy()
    
    # expert = airl_trainer.loadExpertPolicy()
    # airl_trainer.generateRollouts(expert=expert, save_file=True)
    
    airl_trainer.loadRollouts(human=True, filter_stationary=True) # if want to load from file instead of generating

    ### train AIRL
    # airl_trainer.trainAIRL()
    # airl_trainer.evaluatePolicy(policy=airl_trainer.airl_learner)


    ####################################################
    ### 4. Train the main agent using the RNN embedding and modified reward
    ####################################################
    _ = airl_trainer.getDiscriminator()
    modified_agent = airl_trainer.trainModifiedAgent()
    airl_trainer.evaluatePolicy(policy=modified_agent)
    

