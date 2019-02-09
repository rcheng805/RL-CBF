%% Plot Max Deviation from the Safe Set for each Algorithm

clear all
N = 1;
N1 = 10;
load('DDPG-CBF/data1_19-02-08-21-16')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_1(i) = min(a,b);
end
load('DDPG-CBF/data2_19-02-08-21-18')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_2(i) = min(a,b);
end
load('DDPG-CBF/data3_19-02-08-21-17')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_3(i) = min(a,b);
end
load('DDPG-CBF/data4_19-02-08-21-17')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_4(i) = min(a,b);
end

coll_ddpgcbf_min = mean([coll_1; coll_2; coll_3; coll_4]);
coll_ddpgcbf_std = std([coll_1; coll_2; coll_3; coll_4]);

clear coll_1 coll_2 coll_3 coll_4
load('DDPG/data1_19-02-08-17-26')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_1(i) = min(a,b);
end
load('DDPG/data2_19-02-08-17-27')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_2(i) = min(a,b);
end
load('DDPG/data3_19-02-08-17-27')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_3(i) = min(a,b);
end
load('DDPG/data4_19-02-08-17-26')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_4(i) = min(a,b);
end

coll_ddpg_min = mean([coll_1; coll_2; coll_3; coll_4]);
coll_ddpg_std = std([coll_1; coll_2; coll_3; coll_4]);

clear coll_1 coll_2 coll_3 coll_4 data
load('TRPO-CBF/data1_19-02-09-05-13')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_1(i) = min(a,b);
end
load('TRPO-CBF/data2_19-02-09-05-10')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_2(i) = min(a,b);
end
load('TRPO-CBF/data3_19-02-09-05-12')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_3(i) = min(a,b);
end
load('TRPO-CBF/data4_19-02-09-05-07')
for i = 1:length(data)
    a = min(abs(data{i}.Observation(:,7) - data{i}.Observation(:,10)));
    b = min(abs(data{i}.Observation(:,10) - data{i}.Observation(:,13)));
    coll_4(i) = min(a,b);
end

coll_trpocbf_min = mean([coll_1; coll_2; coll_3; coll_4]);
coll_trpocbf_std = std([coll_1; coll_2; coll_3; coll_4]);

clear data
clear coll_1 coll_2 coll_3 coll_4
load('TRPO/data1_19-02-08-11-55')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_1(i) = min(a,b);
end
load('TRPO/data2_19-02-08-11-54')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_2(i) = min(a,b);
end
load('TRPO/data3_19-02-08-11-55')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_3(i) = min(a,b);
end
load('TRPO/data4_19-02-08-11-55')
for i = 1:length(data)
    a = min(data{i}.Observation(:,7) - data{i}.Observation(:,10));
    b = min(data{i}.Observation(:,10) - data{i}.Observation(:,13));
    coll_4(i) = min(a,b);
end

coll_trpo_min = mean([coll_1; coll_2; coll_3; coll_4]);
coll_trpo_std = std([coll_1; coll_2; coll_3; coll_4]);

figure;
hold on
plot(coll_trpocbf_min,'r','LineWidth',1.1)
plot(coll_trpo_min,'b--','LineWidth',1.1)
plot(coll_ddpgcbf_min,'Color',[0,1,0],'LineWidth',1.1)
plot(coll_ddpg_min,'Color',[1,0.5,0],'LineWidth',1.1)
plot([0,1000],[2,2],'k--','LineWidth',2)
xlim([0,400]); xlabel('Episode'); ylabel('Minimum Distance b/n Cars')
ylim([-2,4]); legend('TRPO-CBF','TRPO','DDPG-CBF','DDPG','Safety Boundary'); 
set(gca,'FontSize',16); title('Car Collisions')
hold off

figure;
hold on
shadedErrorBar(1:length(coll_trpo_min), coll_trpo_min, coll_trpo_std, 'lineProps', {'Color','b','LineWidth',1.1})
shadedErrorBar(1:length(coll_trpocbf_min), coll_trpocbf_min, coll_trpocbf_std, 'lineProps', {'Color','r','LineWidth',1.1})
shadedErrorBar(1:length(coll_ddpg_min), coll_ddpg_min, coll_ddpgcbf_std, 'lineProps', {'Color',[0,1,0],'LineWidth',1.1})
shadedErrorBar(1:length(coll_ddpgcbf_min), coll_ddpgcbf_min, coll_ddpgcbf_std, 'lineProps', {'Color',[1,0.5,0],'LineWidth',1.1})
plot([0,1000],[2,2],'k--','LineWidth',2)
xlim([0,400]); xlabel('Episode'); ylabel('Minimum Distance b/n Cars')
ylim([-8,4]); legend('TRPO','TRPO-CBF','DDPG','DDPG-CBF','Safety Boundary'); 
set(gca,'FontSize',16); title('Car Collisions')
hold off