mkdir -p output

# RL4RS environment
EXP_PREFIX="multimodel"


mkdir -p output/${EXP_PREFIX}/
mkdir -p output/${EXP_PREFIX}/agent

env_path="output/kuairand/env/"
output_path=output/${EXP_PREFIX}/agent/

# environment arguments
ENV_CLASS='KRCrossSessionEnvironment_GPU'
MAX_SESSION=10
TEMPER=10

EXP_DEFAULT_NAME="adarec"
EXP=${EXP:-$EXP_DEFAULT_NAME}

# policy arguments
POLICY_CLASS='PAURActionTransformer'

# critic arguments
CRITIC_CLASS='QCritic'

# agent arguments
GAMMA=0.0
N_ITER=50000
EP_BS=128
BS=256
INITEP=0.0
ELBOW=0.1

# facade arguments
NOISE=0.01
REWARD_FUNC='get_retention_reward'
BETA=30
DELTA=30
REG=0.00001
AGENT=PAUR
AUX_WEIGHT=0.003
AUX_TYPE=Q
RLUR_GAMMA=True
PROB_MULTI=1.0
PROB_HIGH=1.2
PROB_LOW=0.8
CRITIC_LR=0.001
ACTOR_LR=0.0001

for SEED in 5 10 15 20 25
do
    file_key=${EXP}_${POLICY_CLASS}_PROB${PROB_HIGH}_${PROB_LOW}_AUX${AUX_WEIGHT}${AUX_TYPE}_RLURGAMMA${RLUR_GAMMA}_beta${BETA}_delta${DELTA}_actor${ACTOR_LR}_critic${CRITIC_LR}_agent${AGENT}_reg${REG}_ep${INITEP}_noise${NOISE}_seed${SEED}
    date_key=$(date +"%m_%d_%H_%M_%S")
    
    mkdir -p ${output_path}/${file_key}/${date_key}/

    python train_adarec.py\
        --filename $0\
        --response_prob_high ${PROB_HIGH}\
        --response_prob_low ${PROB_LOW}\
        --rlur_gamma ${RLUR_GAMMA}\
        --aux_loss_weight ${AUX_WEIGHT}\
        --aux_loss_type ${AUX_TYPE}\
        --seed ${SEED}\
        --cuda 0\
        --env_class ${ENV_CLASS}\
        --uirm_log_path ${env_path}log/user_KRMBUserResponse_MaxOut_lr0.0001_reg0.model.log\
        --urrm_log_path ${env_path}log/user_KRUserRetention_NoSeq_Im_lr0.000001_reg0.model.log\
        --max_n_session ${MAX_SESSION}\
        --initial_temper ${TEMPER}\
        --slate_size 6\
        --policy_class ${POLICY_CLASS}\
        --user_latent_dim 16\
        --item_latent_dim 16\
        --enc_dim 32\
        --attn_n_head 4\
        --anneal_a_lr True\
        --transformer_d_forward 64\
        --transformer_n_layer 2\
        --action_hidden_dims 128\
        --dropout_rate 0.1\
        --critic_class ${CRITIC_CLASS}\
        --critic_hidden_dims 128 32\
        --critic_dropout_rate 0.1\
        --agent_class ${AGENT}\
        --actor_lr ${ACTOR_LR}\
        --critic_lr ${CRITIC_LR}\
        --actor_decay ${REG}\
        --critic_decay ${REG}\
        --target_mitigate_coef 0.01\
        --gamma ${GAMMA}\
        --n_iter ${N_ITER}\
        --episode_batch_size ${EP_BS}\
        --batch_size ${BS}\
        --train_every_n_step 1\
        --initial_greedy_epsilon ${INITEP}\
        --final_greedy_epsilon 0.0\
        --elbow_greedy ${ELBOW}\
        --check_episode 100\
        --save_episode 1000\
        --save_path ${output_path}/${file_key}/${date_key}/model\
        --facade_class AdaRecFacade\
        --beta ${BETA}\
        --delta ${DELTA}\
        --buffer_size 50000\
        --start_train_at_step 2000\
        --noise_var ${NOISE}\
        --noise_clip 1\
        --reward_func ${REWARD_FUNC}\
        | tee ${output_path}/${file_key}/${date_key}/log
done
