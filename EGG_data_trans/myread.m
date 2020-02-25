%% ʹ��˵����
%% 1.������biosig_installer.m,Ȼ�������д˳���
%% 2.�˳���ŵ�BCICIV_2a_gdf��ͬĿ¼������
clear
clc
rootdir='BCICIV_2a_gdf\';
subdir=dir(rootdir);
len=length(subdir);
saldir = '..\data\';
mkdir(saldir)
for i=3:len
    subdirpath=fullfile(rootdir,subdir(i).name);
    [s,HDR]=sload(subdirpath);
    %clear trim1
    %trim= HDR.TRIG;%��ȡ288��Ƭ��
    %trim1=[trim1(1)-2000;trim1];
    %image=zeros(1125,25,288);
    %type=zeros(1125,1);
    %for j=1:288
        %image(:,:,j)=s(trim1(j):trim1(j)+1125-1,:); 
        %type(j,1)=HDR.Classlabel(j);
    %end   
    %trim1=[trim1(1);trim1+2000];
    
    
    image=zeros(1125,25,288);
    type=zeros(1125,1);
    for j=1:288
        image(:,:,j)=s(HDR.TRIG(j)+1.5*250:HDR.TRIG(j)+6*250-1,:);
        type(j,1)=HDR.Classlabel(j);
        
    end
   
    f=regexp(subdir(i).name,'\.','split');
    filename=f{1};
    savePath = [saldir,filename, '_slice' '.mat'];
    save(savePath,'image','type');  % ���浽�ļ���
end