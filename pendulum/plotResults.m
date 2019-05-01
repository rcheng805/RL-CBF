%% Plot Accumulated Reward at each Episode for each Algorithm

clear all
N = 1;
N1 = 10;
load('DDPG-CBF/data1_19-02-08-18-41')
for i = 1:length(data)
    reward_1(i) = sum(data{i}.Reward);
end
reward_1 = movmean(reward_1,N);
load('DDPG-CBF/data2_19-02-08-18-54')
for i = 1:length(data)
    reward_2(i) = sum(data{i}.Reward);
end
reward_2 = movmean(reward_2,N);
load('DDPG-CBF/data3_19-02-08-18-56')
for i = 1:length(data)
    reward_3(i) = sum(data{i}.Reward);
end
reward_3 = movmean(reward_3,N);
load('DDPG-CBF/data4_19-02-08-18-59')
for i = 1:length(data)
    reward_4(i) = sum(data{i}.Reward);
end
reward_4 = movmean(reward_4,N);

reward_mean = movmean(mean([reward_1; reward_2; ...
    reward_3; reward_4]), N1);
reward_std = movmean(std([reward_1; reward_2; ...
    reward_3; reward_4]), N1);

clear data reward_1 reward_2 reward_3 reward_4
T = 600;
load('DDPG/data1_19-02-08-02-12')
reward_1 = movmean(reward(1:T),N);
load('DDPG/data2_19-02-08-02-11')
reward_2 = movmean(reward(1:T),N);
load('DDPG/data3_19-02-08-02-12')
reward_3 = movmean(reward(1:T),N);
load('DDPG/data4_19-02-08-02-13')
reward_4 = movmean(reward(1:T),N);

reward_mean_ddpg = movmean(mean([reward_1; reward_2; reward_3; reward_4]), N1);
reward_std_ddpg = movmean(std([reward_1; reward_2; reward_3; reward_4]), N1);

clear data reward_1 reward_2 reward_3 reward_4
load('TRPO-CBF/data1_19-02-09-04-24')
for i = 1:length(data)
    reward_1(i) = sum(data{i}.Reward)/18;
end
reward_1 = movmean(reward_1,N);
load('TRPO-CBF/data2_19-02-09-04-37')
for i = 1:length(data)
    reward_2(i) = sum(data{i}.Reward)/18;
end
reward_2 = movmean(reward_2,N);
load('TRPO-CBF/data3_19-02-09-04-12')
for i = 1:length(data)
    reward_3(i) = sum(data{i}.Reward)/18;
end
reward_3 = movmean(reward_3,N);
load('TRPO-CBF/data4_19-02-09-04-58')
for i = 1:length(data)
    reward_4(i) = sum(data{i}.Reward)/18;
end
reward_4 = movmean(reward_4,N);

reward_mean_trpo_cbf = movmean(mean([reward_1; reward_2; reward_3; ...
    reward_4]), N1);
reward_std_trpo_cbf = movmean(std([reward_1; reward_2; reward_3; ...
    reward_4]), N1);



%% Load TRPO Data
clear data reward_1 reward_2 reward_3 reward_4
dinfo = dir('TRPO/*.mat');
simdata = cell(length(dinfo),1);
reward = cell(length(dinfo),1);
reward_movmean = cell(length(dinfo),1);
for j = 1:length(dinfo)
    thisfilename = dinfo(j).name;  %just the name
    simdata{j} = load(thisfilename); %load just this file    
    reward{j} = zeros(length(simdata{j}));
    for i = 1:length(simdata{j}.data)
        reward{j}(i) = sum(simdata{j}.data{i}.Reward)/18;
    end
    reward_movmean{j} = movmean(reward{j},N);
end
reward_array = zeros(length(dinfo),length(reward_movmean{j}));
for j = 1:length(dinfo)
    reward_array(j,:) = reward_movmean{j};
end
reward_mean_trpo = movmean(mean(reward_array), N1);
reward_std_trpo = movmean(std(reward_array), N1);

% Plot TRPO Reward
figure;
hold on
plot(0,0,'r'); plot(0,0,'b--');
shadedErrorBar(1:length(reward_mean_trpo), reward_mean_trpo, reward_std_trpo,'lineProps','b--')

load('TRPO/data1_19-02-08-01-52')
for i = 1:length(data)
    reward_1(i) = sum(data{i}.Reward)/18;
end
reward_1 = movmean(reward_1,N);
load('TRPO/data2_19-02-08-01-57')
for i = 1:length(data)
    reward_2(i) = sum(data{i}.Reward)/18;
end
reward_2 = movmean(reward_2,N);
load('TRPO/data3_19-02-08-01-55')
for i = 1:length(data)
    reward_3(i) = sum(data{i}.Reward)/18;
end
reward_3 = movmean(reward_3,N);
load('TRPO/data4_19-02-08-01-52')
for i = 1:length(data)
    reward_4(i) = sum(data{i}.Reward)/18;
end
reward_4 = movmean(reward_4,N);

reward_mean_trpo = movmean(mean([reward_1; reward_2; reward_3; reward_4]), N1);
reward_std_trpo = movmean(std([reward_1; reward_2; reward_3; reward_4]), N1);


figure;
hold on
plot(0,0,'r'); plot(0,0,'b--');
shadedErrorBar(1:length(reward_mean), reward_mean, reward_std,'lineProps','r')
shadedErrorBar(1:length(reward_mean_ddpg), reward_mean_ddpg, reward_std_ddpg,'lineProps','b--')
% shadedErrorBar(1:length(reward_mean_trpo_cbf), reward_mean_trpo_cbf, reward_std_trpo_cbf,'lineProps','b--')
% shadedErrorBar(1:length(reward_mean_trpo), reward_mean_trpo, reward_std_trpo,'lineProps','c')
hold off
xlim([0,400]); xlabel('Episode'); ylim([-1000,100]); ylabel('Reward')
title('Pendulum Reward'); legend('ddpg-cbf','ddpg','Location','southeast')
set(gca,'FontSize',16)

figure;
hold on
plot(0,0,'r'); plot(0,0,'b--');
% shadedErrorBar(1:length(reward_mean), reward_mean, reward_std,'lineProps','r')
% shadedErrorBar(1:length(reward_mean_ddpg), reward_mean_ddpg, reward_std_ddpg,'lineProps','g--')
shadedErrorBar(1:length(reward_mean_trpo_cbf), reward_mean_trpo_cbf, reward_std_trpo_cbf,'lineProps','r')
shadedErrorBar(1:length(reward_mean_trpo), reward_mean_trpo, reward_std_trpo,'lineProps','b--')
hold off
xlim([0,600]); xlabel('Episode'); ylim([-1000,100]); ylabel('Reward')
title('Pendulum Reward'); legend('trpo-cbf','trpo','Location','southeast')
set(gca,'FontSize',16)


figure;
hold on
plot(0,0,'r'); plot(0,0,'g--'); plot(0,0,'b--'); plot(0,0,'c');
shadedErrorBar(1:length(reward_mean), reward_mean, reward_std,'lineProps','r')
shadedErrorBar(1:length(reward_mean_ddpg), reward_mean_ddpg, reward_std_ddpg,'lineProps','g--')
shadedErrorBar(1:length(reward_mean_trpo_cbf), reward_mean_trpo_cbf, reward_std_trpo_cbf,'lineProps','b--')
shadedErrorBar(1:length(reward_mean_trpo), reward_mean_trpo, reward_std_trpo,'lineProps','c')
hold off
xlim([0,400]); xlabel('Episode'); ylim([-1000,100]); ylabel('Reward')
title('Pendulum Reward'); legend('ddpg-cbf','ddpg','trpo-cbf','trpo','Location','southeast')
set(gca,'FontSize',16)