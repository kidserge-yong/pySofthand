data_name = "train_data_09_17_2020_11_13_54.csv";
table = readtable("data/"+data_name);

emg = table{:, 1:32};
angle = table{:, 33:end};

n = size(emg,1);
m = size(emg,2);
Syn_num = 3;

U = rand(n,Syn_num);
V = rand(m,Syn_num);
[U,V,CPUtime,NRV,RRV] = FastHALS(emg, U, V, 1000);
result = FastCal(emg, V, 30);
stackedplot(result);