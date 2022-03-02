# -PCA-Eigenfaces-LDA-Fisherfaces-
本次实验主要是学习PCA，并基于PCA进行各种操作，此外有附加的LDA降维与PCA的对比分析以及人脸识别算法设计。
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVMAH.md.png"></div>
  
## 流程
①利用numpy实现PCA模型。  
②利用PCA模型对testSet.txt数据集进行降维，绘制出降维后图像。  
③对secom.data数据集进行降维，分析主成分数为多少时最优。  
④实现基于PCA的Eigenfaces算法，并利用留出法验证。
⑤实现基于LDA的Fisherfaces算法，并利用留出法验证。
⑥人脸检测算法设计
## 内容
### 1. numpy实现PCA并运用于testSet.txt数据集
PCA的实现可从以下步骤进行：  
①对数据去中心化  
②计算数据的协方差矩阵  
③计算协方差矩阵的特征向量和特征值  
④选取最大的K个特征值对应的特征向量作为降维矩阵的基  
对应到numpy中，我们要用到以下几个函数：  
①numpy.mean 求平均值，参数axis = 1表示对行，axis = 0对列。  
②numpy.cov 求矩阵的协方差，当然，我们也可以根据定义来求，即使用XX^T/(m-1)，其中m表示特征数。  
③np.linalg.eig 返回特征值和特征向量，这里的特征向量是已经标准化的。  
④numpy.argsort 返回从小到大的下标  
我们通过利用这几个函数便可以很方便的实现PCA模型。  
有了PCA的模型，我们接下来就是获取数据，对于testSet.txt而言，数据格式非常简单，我们通过split函数可以轻松的实现特征值的分离，保存为float格式便得到了数据。之后将数据传入PCA模型，得到降维后的坐标，为了更好的显示降维效果，我们可以将降维后的值重新转化到原特征空间进行显示。  
最后利用plt库进行图形的显示  
### 2. 对半导体数据进行降维处理
对半导体数据secom.data的降维与对testSet.txt的降维在本质上没有什么区别，都是利用PCA找到最大的K个特征值和特征向量。  
但是，testSet.txt的数据比较特殊，他只有两维，所以如果降维的话一定是降到一维，没有所谓的主成分数选择多少最好的问题。在secom.data数据集中，每个数据包含了590个特征，我们降维的维度就有了很多个选择，因此，对半导体数据的降维除了利用PCA之外，还包括对主成分数的选取。  
至于降维，就是PCA。对于主成分数目的选取来说，我们可以通过选取不同的主成分数目多次实验，对降维效果进行对比，做出最后的数目选择。在本实验中，我们对降维效果的评价标准是降维后方差的占比。  
对于本数据集比较特殊的一点是，数据中存在大量的NAN数据，即无标记数据。对于每一个数据，我们利用已知特征值的平均代替本向量的缺失值。  
所以，最后的实验内容记为，对数据集进行导入，并对NAN数据进行缺失平均替代处理。之后对完整数据集进行PCA，得到所有的特征值和特征向量之后，选取不同的主成分数目进行降维。分析不同主成分数对应的效果，得到最后的主成分数。  
### 3. 实现基于PCA的Eigenfaces算法
Yale数据集包含了165张照片，为15个人的11种状态，本次实验基于此数据集进行操作。  
首先，本实验的数据集比较特殊，不是常见的jpg或者png格式的图片，而是pgm。Pgm图片分为不同的模式，有P2、P5等。这些模式的读取方式不同，对于P4-P6的模式，可以使用PIL非常方便的进行读取。本次实验的pmg图片为P5模式，可利用NotePad++查看，  
<div align=center><img src = "https://s4.ax1x.com/2022/03/02/bGVnBD.png"></div>  
<div align=center>Yale数据集pmg图片格式查看</div>  
  
所以，本次实验的数据可以直接利用Image.open()直接导入，之后利用numpy转化为numpy数据即可。而对标签的获取，可以直接利用名字进行识别，取诸如“subject01”等作为对应人标签即可。  
至此，我们对数据进行了导入，接下来便是利用Eigenfaces(特征脸)算法进行操作。  
特征脸算法可分为以下几个步骤取操作：  
①将二维图片矩阵reshape为一维向量  
②将所有一维向量合并为一个矩阵，记为A  
③计算所有向量的平均值（此时我们将得到一个平均脸，可以输出看一看）  
④将步骤二得到的矩阵A去中心化，即每个向量减去平均值  
⑤按一般步骤来说，我们这步可以计算AA^T的特征值和特征向量。但是一般由于这个过大，我们可以通过计算A^TA的特征值和特征向量去反推AA^T的特征值和特征向量。  
⑥得到降维特征矩阵W  
⑦将已知的数据降维到W对应特征空间中，组成一个新的特征数据集。  
⑧对于新数据，利用W将其转化到此特征空间，利用聚类算法识别其对应标签。（论文中为最短欧氏距离）  
根据project-4.pdf中的要求，我分别进行了三项实验，对应取N为3、5、7。每次实验中降维空间K的取值分布为1~100.每次实验重复10次进行，最后误差取平均值。  
### 4. 实现基于LDA的fisherfaces算法
LDA是一种监督学习的降维技术，也就是说它的数据集的每个样本是有类别输出的。这点和PCA不同。PCA是不考虑样本类别输出的无监督降维技术。LDA的思想可以用一句话概括，就是“投影后类内方差最小，类间方差最大”。  
fisherfaces算法可分为以下几个步骤取操作：  
① 计算类内散度矩阵  
② 计算类间散度矩阵  
③ 计算矩阵  
④ 计算的最大的d个特征值和对应的d个特征向量,得到投影矩阵  
⑤ 对样本集中的每一个样本特征,转化为新的样本  
⑥ 得到输出样本集  
## 结果
### 1. numpy实现PCA并运用于testSet.txt数据集
对testSet的降维结果如图3.1.1所示，其中蓝色散点为原数据在二维空间中对应位置，橘色散点为降维后的散点在原数据空间对应位置。 
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVmnO.png"></div>  
<div align=center>testSet数据集降维结果 </div>  
    
通过图片，我们可以大致感受到，PCA对于降维的效果是让点投影更散，这也是PCA最大方差和最小间隔的体现。  
比较有趣的是，本数据集有比较明显的方向性，如果我们取最小的几个主成分作为特征向量，那么效果将会如图所示。
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVZjK.png"></div>
<div align=center>最差投影效果  </div>   
    
所以，PCA的降维显得比较直观，比较好理解，这也是其突出的优点之一。  
### 2. 对半导体数据进行降维处理
不同于testSet数据集，secom数据集的降维效果没有办法直观的显示，所以只能以其他方法替代，下边两幅图为从大到小特征值对应方差百分比的曲线图。  
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVVc6.png"></div>
<div align=center>从大到小的特征值对应方差</div> 
     
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVuHe.png"></div>
<div align=center>从大到小特征值积累方差</div>
     
具体而言，第一个主成分占了59%，第二个占了23%，第三个为8.8%，并以指数衰减，到第19个时，数量级已经降为10的负4次方。前八个主成分已经占了97%的比例，前18个已经占了99%的比例。  
考虑到复杂度和精确度来说，我们选取8~18个已经足够了。  
### 3. 实现基于PCA的Eigenfaces算法
N分别取3、5、7时的平均脸如图所示。可以发现平均脸的变化并不是很大。
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGV39I.md.png"></div>
<div align=center>N分别取3、5、7时的平均脸</div>
    
除了平均脸之外，我们通过对每个数据进行降维还可以得到一个特征脸谱图。直观上来看，特征脸谱图同样与N的关系不大，但是特征脸谱图与主成分数目选取关系较为大，下边为对比图：  
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVl4A.png"></div>
<div align=center>主成分数目为10的特征脸谱图 </div>
   
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVQNd.png"></div>
<div align=center>主成分数目为30的特征脸谱图</div>
    
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGV83t.png"></div>
<div align=center>主成分数目为50的特征脸谱图</div>
    
可以发现，当主成分数增加时，我们的轮廓显得更加的清晰，对特征的提取更加的明显。
下边为N分别取3、5、7时，主成分从1~100变化的10次平均误差变化图：
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVGgP.png"></div>
<div align=center>N为3时的误差图</div>
    
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVJjf.png"></div>
<div align=center>N为5时的误差图</div>
   
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVtu8.png"></div>
<div align=center>N为7时的误差图</div>  
    
可以发现，综合来看，K并不是越大越好，当主成分数目取到10~30时，分类的效果达到最好，当主成分数增加到很多时，效果反而下降。
### 4. 实现基于LDA的fisherfaces算法
LDA的K取了1~20:
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVNDS.png"></div>
<div align=center>N为3时的误差图</div>
  
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVUHg.png"></div>
<div align=center>N为5时的误差图</div>
    
<div align=center><img src ="https://s4.ax1x.com/2022/03/02/bGVdEQ.png"></div>
<div align=center>N为7时的误差图</div>
