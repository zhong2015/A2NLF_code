package recommender;

import recommender.common.CommonRec;
import recommender.common.RTuple;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class ANLF extends CommonRec {

    public ANLF(){
        super();
    }

    public static void main(String[] args) throws IOException {

        CommonRec.dataSetName = "EM";
        String filePath = "/Users/zhong/Desktop/A2NLF/DS/";
        CommonRec.dataLoad(filePath + CommonRec.dataSetName + "_train.txt", filePath + CommonRec.dataSetName+ "_val.txt","::");
        System.out.println("Max User ID:\t" + CommonRec.userMaxID);
        System.out.println("Max Item ID:\t" + CommonRec.itemMaxID);
        System.out.println("训练集的容量："+ CommonRec.trainDataSet.size());
        System.out.println("测试集的容量："+ CommonRec.testDataSet.size());

        // 设置公共参数
        CommonRec.maxRound = 1000;
        CommonRec.minGap = 1e-5;
        CommonRec.delayCount = 5;

        for(int tempdim = 20; tempdim <= 20; tempdim += CommonRec.featureDimension){

            CommonRec.featureDimension = tempdim;
            CommonRec.userXFeatureSaveDir = "./savedLFs/"+ CommonRec.dataSetName +"/X";
            CommonRec.itemYFeatureSaveDir = "./savedLFs/"+ CommonRec.dataSetName +"/Y";

            // 初始化特征矩阵
            CommonRec.initStaticFeatures();

            experimenter(CommonRec.RMSE);
            experimenter(CommonRec.MAE);
        }
    }

    public static void experimenter(int metrics) throws IOException {

        long file_tMills = System.currentTimeMillis(); //用于给train函数打开在当前函数所创建的文件
        FileWriter fw;
        if(metrics == CommonRec.RMSE)
            fw = new FileWriter(new File("./" + CommonRec.dataSetName + "_RMSE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "dim=" + featureDimension + ".txt"), true);
        else
            fw = new FileWriter(new File("./" + CommonRec.dataSetName + "_MAE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "dim=" + featureDimension + ".txt"), true);

        String blankStr = "                          ";
        String starStr = "****************************************************************";
        String equalStr = "=====================================";

        for (double tempLam = Math.pow(2,-4); tempLam <= Math.pow(2,2); tempLam *= 2) {

            CommonRec.lambda = tempLam;

            // 打印标题项
            System.out.println("\n" + starStr);
            System.out.println(blankStr + "featureDimension——>" + CommonRec.featureDimension);
            System.out.println(blankStr + "lambda——>" + CommonRec.lambda);
            System.out.println(blankStr + "minGap——>" + CommonRec.minGap);
            System.out.println(starStr);

            fw.write("\n" + starStr + "\n");
            fw.write(blankStr + "featureDimension——>" + CommonRec.featureDimension + "\n");
            fw.write(blankStr + "lambda——>" + CommonRec.lambda + "\n");
            fw.write(blankStr + "minGap——>" + CommonRec.minGap + "\n");
            fw.write(starStr + "\n");
            fw.flush();

            // 按学习率eta取值的不同进行测试
            for(double tempEta = 1; tempEta <= 4; tempEta += 0.5){

                CommonRec.eta = tempEta;

                fw.write("\n" + equalStr + "\n");
                fw.write("        Eta——>" + CommonRec.eta + "\n");
                fw.write(equalStr + "\n");
                fw.flush();

                System.out.println("\n" + equalStr);
                System.out.println("        Eta——>" + CommonRec.eta);
                System.out.println(equalStr);

                // 为确保每一次gamma取新值后是一个新的更新过程，则每次重新创建一个MSNLF对象，这些对象的参数取值是一致的
                ANLF trainANLF = new ANLF();
//                trainANLF.printNegativeFeature(); // 检查是否有负初始值
                // 开始训练
                trainANLF.train(metrics, fw);

                System.out.println("Min training Error:\t\t\t" + trainANLF.min_Error);
                System.out.println("Min total training epochs:\t\t" + trainANLF.min_Round);
                System.out.println("Total Round:\t\t" + trainANLF.total_Round);
                System.out.println("Min total training time:\t\t" + trainANLF.minTotalTime);
                System.out.println("Min average training time:\t\t" + trainANLF.minTotalTime / trainANLF.min_Round);
                System.out.println("Total training time:\t\t" + trainANLF.total_Time);
                System.out.println("Average training time:\t\t" + trainANLF.total_Time / trainANLF.total_Round);

                fw.write("Min training Error:\t\t\t" + trainANLF.min_Error + "\n");
                fw.write("Min total training epochs:\t\t" + trainANLF.min_Round + "\n");
                fw.write("Total Round:\t\t" + trainANLF.total_Round + "\n");
                fw.write("Min total training time:\t\t" + trainANLF.minTotalTime + "\n");
                fw.write("Min average training time:\t\t" + trainANLF.minTotalTime / trainANLF.min_Round + "\n");
                fw.write("Total training time:\t\t" + trainANLF.total_Time + "\n");
                fw.write("Average training time:\t\t" + trainANLF.total_Time / trainANLF.total_Round + "\n");
                fw.flush();
            }
        }
        fw.close();
    }

    public void train(int metrics, FileWriter fw) throws IOException {

        double lastErr = 0;

        // 初始化：将所有的rating估计值缓存起来，提高计算效率
        for (RTuple trainR : trainDataSet) {
            double ratingHat = dotMultiply(X[trainR.userID], Y[trainR.itemID]);
            trainR.ratingHat = ratingHat;
        }

//        outPutModelNonnega(0, 0);

        for (int round = 1; round <= maxRound; round++) {

            double startTime = System.currentTimeMillis();
            double rho, tau;
            for (int dim = 0; dim < featureDimension; dim++) {
                // 将相关的辅助变量置为0
                resetAuxArray();

                for (RTuple trainR : trainDataSet) {
                    // 对于X，记录学习增量
                    // 分别计算对于用户的上下更新
                    X_U[trainR.userID] -= Y[trainR.itemID][dim] * (trainR.ratingHat - trainR.rating);
                    X_U[trainR.userID] += X[trainR.userID][dim] * Y[trainR.itemID][dim]
                            * Y[trainR.itemID][dim];
                    X_D[trainR.userID] += Y[trainR.itemID][dim] * Y[trainR.itemID][dim];
                    // 对于Y，记录学习增量
                    // 分别计算对于商品的上下更新
                    Y_U[trainR.itemID] -= X[trainR.userID][dim] * (trainR.ratingHat - trainR.rating);
                    Y_U[trainR.itemID] += Y[trainR.itemID][dim] * X[trainR.userID][dim]
                            * X[trainR.userID][dim];
                    Y_D[trainR.itemID] += X[trainR.userID][dim] * X[trainR.userID][dim];
                }

                for (int userID = 1; userID <= userMaxID; userID++){

                    X_C[userID] = X[userID][dim];

                    rho = lambda * userRSetSize[userID];
                    X_U[userID] += rho * P[userID][dim];
                    X_U[userID] -= Gamma_X[userID][dim];
                    if(rho != 0)
                        X_D[userID] += rho;
                    else
                        X_D[userID] += 1e-8;
                    X[userID][dim] = X_U[userID] / X_D[userID];

                    // 更新P矩阵
                    double tempP;
                    if(rho != 0)
                        tempP = X[userID][dim] + Gamma_X[userID][dim] / (rho);
                    else
                        tempP = X[userID][dim] + Gamma_X[userID][dim] / (1e-8);
                    if (tempP > 0)
                        this.P[userID][dim] = tempP;
                    else
                        this.P[userID][dim] = 0;
                    // 更新Gamma_X矩阵
                    Gamma_X[userID][dim] += eta * rho * (X[userID][dim] - P[userID][dim]);
                }

                for (int itemID = 1; itemID <= itemMaxID; itemID++) {

                    Y_C[itemID] = Y[itemID][dim];

                    tau = lambda * itemRSetSize[itemID];
                    Y_U[itemID] += tau * Q[itemID][dim];
                    Y_U[itemID] -= Gamma_Y[itemID][dim];
                    if(tau != 0)
                        Y_D[itemID] += tau;
                    else
                        Y_D[itemID] += 1e-8;
                    Y[itemID][dim] = Y_U[itemID] / Y_D[itemID];

                    // 更新Q矩阵
                    double tempQ;
                    if(tau != 0)
                        tempQ = Y[itemID][dim] + Gamma_Y[itemID][dim] / (tau);
                    else
                        tempQ = Y[itemID][dim] + Gamma_Y[itemID][dim] / (1e-8);
                    if (tempQ > 0)
                        Q[itemID][dim] = tempQ;
                    else
                        Q[itemID][dim] = 0;
                    // 更新Gamma_Y矩阵
                    Gamma_Y[itemID][dim] += eta * tau * (Y[itemID][dim] - Q[itemID][dim]);
                }

                // 根据X,Y矩阵的更新，对所有的rating估计值进行更新
                for (RTuple trainR : trainDataSet) {
                    double ratingHatNew = X[trainR.userID][dim] * Y[trainR.itemID][dim]
                            - X_C[trainR.userID] * Y_C[trainR.itemID];
                    trainR.ratingHat = trainR.ratingHat + ratingHatNew;
                }
            }

            double endTime = System.currentTimeMillis();
            cacheTotalTime += endTime - startTime;
            total_Time += endTime - startTime;
            // 计算本轮训练结束后，在测试集上的误差
            double curErr;
            if (metrics == CommonRec.RMSE)
                curErr = testRMSE();
            else
                curErr = testMAE();
            fw.write(curErr + "\n");
            fw.flush();
            System.out.println(curErr);

            total_Round += 1;
            if (min_Error > curErr) {
                min_Error = curErr;
                min_Round = round;
                this.minTotalTime += this.cacheTotalTime;
                this.cacheTotalTime = 0;
            }
            else if ((round - min_Round) >= delayCount) {
                break;
            }

            if (Math.abs(curErr - lastErr) > minGap)
                lastErr = curErr;
            else
                break;
        }
    }
}
