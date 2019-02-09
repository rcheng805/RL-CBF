%% Plot Accumulated Reward at each Episode for each Algorithm

clear all
N = 1;
N1 = 10;
load('DDPG-CBF/data1_19-02-08-21-16')
clear reward
for i = 1:length(data)
    if (iscell(data{i}.Reward))
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_1(i) = sum(reward);
end
reward_1 = movmean(reward_1,N);
load('DDPG-CBF/data2_19-02-08-21-18')
clear reward
for i = 1:length(data)
    if (iscell(data{i}.Reward))
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_2(i) = sum(reward);
end
reward_2 = movmean(reward_2,N);
load('DDPG-CBF/data3_19-02-08-21-17')
clear reward
for i = 1:length(data)
    if (iscell(data{i}.Reward))
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_3(i) = sum(reward);
end
reward_3 = movmean(reward_3,N);
load('DDPG-CBF/data4_19-02-08-21-17')
clear reward
for i = 1:length(data)
    if (iscell(data{i}.Reward))
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_4(i) = sum(reward);
end
reward_4 = movmean(reward_4,N);

reward_mean_ddpg_cbf = movmean(mean([reward_1; reward_2; reward_3; reward_4]), N1);
reward_std_ddpg_cbf = movmean(std([reward_1; reward_2; reward_3; reward_4]), N1);



load('DDPG/data1_19-02-08-17-26')
reward_1 = movmean(reward,N);
load('DDPG/data2_19-02-08-17-27')
reward_2 = movmean(reward,N);
load('DDPG/data3_19-02-08-17-27')
reward_3 = movmean(reward,N);
load('DDPG/data4_19-02-08-17-26')
reward_4 = movmean(reward,N);
reward_mean_ddpg = movmean(mean([reward_1; reward_2; reward_3; reward_4]), N1);
reward_std_ddpg = movmean(std([reward_1; reward_2; reward_3; reward_4]), N1);

load('TRPO-CBF/data1_19-02-09-05-13')
for i = 1:length(data)
    if iscell(data{i}.Reward)
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_1(i) = sum(reward)/15;
end
reward_1 = movmean(reward_1,N);
load('TRPO-CBF/data2_19-02-09-05-10')
for i = 1:length(data)
    if iscell(data{i}.Reward)
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_2(i) = sum(reward)/15;
end
reward_2 = movmean(reward_2,N);
load('TRPO-CBF/data3_19-02-09-05-12')
for i = 1:length(data)
    if iscell(data{i}.Reward)
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_3(i) = sum(reward)/15;
end
reward_3 = movmean(reward_3,N);
load('TRPO-CBF/data4_19-02-09-05-07')
for i = 1:length(data)
    if iscell(data{i}.Reward)
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward{j};
        end
    else
        for j = 1:length(data{i}.Reward)
            reward(j) = data{i}.Reward(j);
        end
    end
    reward_4(i) = sum(reward)/15;
end
reward_4 = movmean(reward_4,N);

reward_mean_trpo_cbf = movmean(mean([reward_1; reward_2; ...
    reward_3; reward_4]), N1);
reward_std_trpo_cbf = movmean(std([reward_1; reward_2; ...
    reward_3; reward_4]), N1);

load('TRPO/data1_19-02-08-11-55')
for i = 1:length(data)
    reward_1(i) = sum(data{i}.Reward)/15;
end
reward_1 = movmean(reward_1,N);
load('TRPO/data2_19-02-08-11-54')
for i = 1:length(data)
    reward_2(i) = sum(data{i}.Reward)/15;
end
reward_2 = movmean(reward_2,N);
load('TRPO/data3_19-02-08-11-55')
for i = 1:length(data)
    reward_3(i) = sum(data{i}.Reward)/15;
end
reward_3 = movmean(reward_3,N);
load('TRPO/data4_19-02-08-11-55')
for i = 1:length(data)
    reward_4(i) = sum(data{i}.Reward)/15;
end

reward_mean_trpo = movmean(mean([reward_1; reward_2; reward_3; reward_4]), N1);
reward_std_trpo = movmean(std([reward_1; reward_2; reward_3; reward_4]), N1);

figure;
hold on
plot(0,0,'r')
plot(0,0,'b--')
plot(0,0,'g')
% plot(0,0,'Color',[1,0.5,0])
shadedErrorBar(1:length(reward_mean_trpo_cbf), reward_mean_trpo_cbf, reward_std_trpo_cbf,'lineProps','r')
shadedErrorBar(1:length(reward_mean_trpo), reward_mean_trpo, reward_std_trpo,'lineProps','b--')
shadedErrorBar(1:length(reward_mean_ddpg_cbf), reward_mean_ddpg_cbf, reward_std_ddpg_cbf,'lineProps','g')
% shadedErrorBar(1:length(reward_mean_ddpg), reward_mean_ddpg, reward_std_ddpg,'lineProps',{'Color',[1,0.5,0]})
hold off
xlim([0,1000]); ylim([-60000,10000]); xlabel('Episode'); ylabel('Reward')
set(gca,'FontSize',16'); legend('TRPO-CBF','TRPO','DDPG-CBF')
title('Car-Following')