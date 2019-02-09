%% Plot Max Deviation from the Safe Set for each Algorithm

clear all
N = 1;
N1 = 2;
load('DDPG-CBF/data1_19-02-08-18-41')
for i = 1:length(data)
    coll_1(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('DDPG-CBF/data2_19-02-08-18-54')
for i = 1:length(data)
    coll_2(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('DDPG-CBF/data3_19-02-08-18-56')
for i = 1:length(data)
    coll_3(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('DDPG-CBF/data4_19-02-08-18-59')
for i = 1:length(data)
    coll_4(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end

coll_mean_ddpgcbf = movmean(mean([coll_1; coll_2; ...
    coll_3; coll_4]), N1);
coll_std_ddpgcbf = movmean(std([coll_1; coll_2; ...
    coll_3; coll_4]), N1);

clear data coll_1 coll_2 coll_3 coll_4
load('DDPG/data1_19-02-08-02-12')
for i = 1:length(data)
    coll_1(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('DDPG/data2_19-02-08-02-11')
for i = 1:length(data)
    coll_2(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('DDPG/data3_19-02-08-02-12')
for i = 1:length(data)
    coll_3(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('DDPG/data4_19-02-08-02-13')
for i = 1:length(data)
    coll_4(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end

coll_mean_ddpg = movmean(mean([coll_1; coll_2; coll_3; coll_4]), N1);
coll_std_ddpg = movmean(std([coll_1; coll_2; coll_3; coll_4]), N1);

clear data coll_1 coll_2 coll_3 coll_4
load('TRPO-CBF/data1_19-02-09-04-24')
for i = 1:length(data)
    coll_1(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('TRPO-CBF/data2_19-02-09-04-37')
for i = 1:length(data)
    coll_2(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('TRPO-CBF/data3_19-02-09-04-12')
for i = 1:length(data)
    coll_3(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('TRPO-CBF/data4_19-02-09-04-58')
for i = 1:length(data)
    coll_4(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end

coll_mean_trpocbf = movmean(mean([coll_1; coll_2; coll_3; ...
    coll_4]), N1);
coll_std_trpocbf = movmean(std([coll_1; coll_2; coll_3; ...
    coll_4]), N1);

clear data coll_1 coll_2 coll_3 coll_4
load('TRPO/data1_19-02-08-01-52')
for i = 1:length(data)
    coll_1(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('TRPO/data2_19-02-08-01-57')
for i = 1:length(data)
    coll_2(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('TRPO/data3_19-02-08-01-55')
for i = 1:length(data)
    coll_3(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end
load('TRPO/data4_19-02-08-01-52')
for i = 1:length(data)
    coll_4(i) = max(abs(atan2(data{i}.Observation(:,2), data{i}.Observation(:,1))));
end

coll_mean_trpo = movmean(mean([coll_1; coll_2; coll_3; coll_4]), N1);
coll_std_trpo = movmean(std([coll_1; coll_2; coll_3; coll_4]), N1);

figure;
hold on
plot(0,0,'r'); plot(0,0,'g--'); plot(0,0,'b--'); plot(0,0,'c');
shadedErrorBar(1:length(coll_mean_ddpgcbf), coll_mean_ddpgcbf, coll_std_ddpgcbf,'lineProps','r')
shadedErrorBar(1:length(coll_mean_ddpg), coll_mean_ddpg, coll_std_ddpg,'lineProps','g--')
shadedErrorBar(1:length(coll_mean_trpocbf), coll_mean_trpocbf, coll_std_trpocbf,'lineProps','b--')
shadedErrorBar(1:length(coll_mean_trpo), coll_mean_trpo, coll_std_trpo,'lineProps','c')
hold off
xlabel('Episode'); ylabel('Reward')
title('Pendulum Reward'); legend('ddpg-cbf','ddpg','trpo-cbf','trpo')
set(gca,'FontSize',16)

figure;
hold on
% plot(0,0,'r'); plot(0,0,'g--'); plot(0,0,'b--'); plot(0,0,'c');
plot(1:length(coll_mean_ddpgcbf),coll_mean_ddpgcbf,'r')
plot(1:length(coll_mean_ddpg),coll_mean_ddpg, 'g--')
plot(1:length(coll_mean_trpocbf),coll_mean_trpocbf, 'b--')
plot(1:length(coll_mean_trpo),coll_mean_trpo,'c')
plot([0,600],[1,1],'k--','LineWidth',1.5)
hold off
xlabel('Episode'); ylabel('Reward'); xlim([0,400])
title('Pendulum Reward'); legend('ddpg-cbf','ddpg','trpo-cbf','trpo','Safe Boundary')
set(gca,'FontSize',16)