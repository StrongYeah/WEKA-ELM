package weka.classifiers.rules;

import java.io.NotSerializableException;
import java.io.Serializable;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;
import no.uib.cipr.matrix.NotConvergedException;
import no.uib.cipr.matrix.SVD;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class ELM extends AbstractClassifier implements
		WeightedInstancesHandler, TechnicalInformationHandler {

	/**
	 * author ye_qiangsheng on 2014/10/03
	 */
	private static final long serialVersionUID = -7834549585915326436L;

	public static class Inverse implements Serializable {// 内部类序列化问题，改成静态类
		/**
		 * 
		 */
		private static final long serialVersionUID = -2444650397825814553L;
		private DenseMatrix A1;
		// private DenseMatrix A2;
		private int m;
		private int n;

		public Inverse(DenseMatrix AD) {
			m = AD.numRows();
			n = AD.numColumns();
			// if(m == n)
			A1 = AD.copy();
			// else
			// A2 = AD.copy();

		}

		// Just the inverse maxtrix
		public DenseMatrix getInverse() {

			DenseMatrix I = Matrices.identity(n);
			DenseMatrix Ainv = I.copy();
			A1.solve(I, Ainv);
			// I.solve(A1, Ainv);
			return Ainv;
		}

		/*
		 * Moore-Penrose generalized inverse maxtrix Theory:Full rank
		 * factorization [U S Vt] = SVD(A) <==> U*S*Vt = A C=U*sqrt(S)
		 * D=sqrt(S)*Vt <==> A=C*D,Full rank factorization MP(A) =
		 * D'*inv(D*D')*inv(C'*C)*C'
		 */
		public DenseMatrix getMPInverse() throws NotConvergedException {
			//System.out.println(" "+m + n);
			SVD svd = new SVD(m, n); // U*S*Vt=A;
			//System.out.println(java.util.Arrays.toString(svd.getS()));
			//System.out.println("test ... ");			
			svd.factor(A1);
			//System.out.println(java.util.Arrays.toString(svd.getS()));
			//System.out.println("test ... ");
			DenseMatrix U = svd.getU(); // m*m
			DenseMatrix Vt = svd.getVt(); // n*n
			double[] s = svd.getS();
			int sn = s.length;
			for (int i = 0; i < sn; i++) {
				s[i] = Math.sqrt(s[i]);
			}
			DenseMatrix S1 = (DenseMatrix) Matrices.random(m, sn);
			S1.zero();
			DenseMatrix S2 = (DenseMatrix) Matrices.random(sn, n);
			S2.zero();
			for (int i = 0; i < s.length; i++) {
				S1.set(i, i, s[i]);
				S2.set(i, i, s[i]);
			}

			DenseMatrix C = new DenseMatrix(m, sn);
			U.mult(S1, C);
			DenseMatrix D = new DenseMatrix(sn, n);
			S2.mult(Vt, D);

			DenseMatrix DD = new DenseMatrix(sn, sn);
			DenseMatrix DT = new DenseMatrix(n, sn);
			D.transpose(DT);
			D.mult(DT, DD);
			Inverse inv = new Inverse(DD);
			DenseMatrix invDD = inv.getInverse();

			DenseMatrix DDD = new DenseMatrix(n, sn);
			DT.mult(invDD, DDD);

			DenseMatrix CC = new DenseMatrix(sn, sn);
			DenseMatrix CT = new DenseMatrix(sn, m);
			C.transpose(CT);
			// (C.transpose()).mult(C, CC);
			CT.mult(C, CC);
			Inverse inv2 = new Inverse(CC);
			DenseMatrix invCC = inv2.getInverse();

			DenseMatrix CCC = new DenseMatrix(sn, m);
			invCC.mult(CT, CCC);

			DenseMatrix Ainv = new DenseMatrix(n, m);
			DDD.mult(CCC, Ainv);
			return Ainv;
		}

		/*
		 * Moore-Penrose generalized inverse maxtrix Theory:Ridge regression
		 * MP(A) = inv((H'*H+lumda*I))*H'
		 */
		public DenseMatrix getMPInverse(double lumda)
				throws NotConvergedException {
			DenseMatrix At = new DenseMatrix(n, m);
			A1.transpose(At);
			DenseMatrix AtA = new DenseMatrix(n, n);
			At.mult(A1, AtA);

			DenseMatrix I = Matrices.identity(n);
			AtA.add(lumda, I);
			DenseMatrix AtAinv = I.copy();
			AtA.solve(I, AtAinv);

			DenseMatrix Ainv = new DenseMatrix(n, m);
			AtAinv.mult(At, Ainv);
			// DDD.mult(CCC, Ainv);
			return Ainv;
		}

		public DenseMatrix checkCD() throws NotConvergedException {
			SVD svd = new SVD(m, n); // U*S*Vt=A;
			svd.factor(A1);
			DenseMatrix U = svd.getU(); // m*m
			DenseMatrix Vt = svd.getVt(); // n*n
			double[] s = svd.getS();
			int sn = s.length;
			// DenseVector S = new DenseVector(s);
			// for (double d : s) {
			// d = Math.sqrt(d);
			// }
			for (int i = 0; i < s.length; i++) {
				s[i] = Math.sqrt(s[i]);
			}

			// System.out.println("length of S: \n"+s.length+"  "+s[0]+" "+s[1]);

			DenseMatrix S1 = (DenseMatrix) Matrices.random(m, sn);
			S1.zero();
			DenseMatrix S2 = (DenseMatrix) Matrices.random(sn, n);
			S2.zero();
			for (int i = 0; i < s.length; i++) {
				S1.set(i, i, s[i]);
				S2.set(i, i, s[i]);
			}

			DenseMatrix C = new DenseMatrix(m, sn);
			U.mult(S1, C);
			DenseMatrix D = new DenseMatrix(sn, n);
			S2.mult(Vt, D);

			DenseMatrix CD = new DenseMatrix(m, n);
			C.mult(D, CD);

			// DenseMatrix CD = new DenseMatrix(m,n);
			// S1.mult(S2, CD);
			return CD;
		}

	}

	public static class Selm implements Serializable {
		/**
		 * 
		 */
		private static final long serialVersionUID = -4579057650893908831L;
		private static DenseMatrix train_set;
		private DenseMatrix test_set;
		private static int numTrainData;
		private int numTestData;
		private static DenseMatrix InputWeight;
		private static float TrainingTime;
		private float TestingTime;
		private static double TrainingAccuracy;
		private double TestingAccuracy;
		private static int Elm_Type;
		private static int NumberofHiddenNeurons;
		private static int NumberofOutputNeurons; // also the number of classes
		private static int NumberofInputNeurons; // also the number of
													// attribution
		private static String func;
		private static int[] label;
		// this class label employ a lazy and easy method,any class must written
		// in
		// 0,1,2...so the preprocessing is required

		// the blow variables in both train() and test()
		private static DenseMatrix BiasofHiddenNeurons;
		private static DenseMatrix OutputWeight;
		private DenseMatrix testP;
		private DenseMatrix testT;
		private static DenseMatrix Y;
		private static DenseMatrix T;

		// 选择预测的属性，和随机种子
		private static int m_classAtt = 9;
		private static int m_seed = 1;

		/**
		 * Construct an ELM
		 * 
		 * @param elm_type
		 *            - 0 for regression; 1 for (both binary and multi-classes)
		 *            classification
		 * @param numberofHiddenNeurons
		 *            - Number of hidden neurons assigned to the ELM
		 * @param ActivationFunction
		 *            - Type of activation function: 'sig' for Sigmoidal
		 *            function 'sin' for Sine function 'hardlim' for Hardlim
		 *            function 'tribas' for Triangular basis function 'radbas'
		 *            for Radial basis function (for additive type of SLFNs
		 *            instead of RBF type of SLFNs)
		 * @throws NotConvergedException
		 */

		public Selm(int elm_type, int numberofHiddenNeurons,
				String ActivationFunction) {

			Elm_Type = elm_type;
			NumberofHiddenNeurons = numberofHiddenNeurons;
			func = ActivationFunction;

			TrainingTime = 0;
			TestingTime = 0;
			TrainingAccuracy = 0;
			TestingAccuracy = 0;
			NumberofOutputNeurons = 1;
		}

		// --------1
		/**
		 * 构造函数
		 * 
		 * @param elm_type
		 *            - 判断是数值型预测，还是名词性分类
		 * @param numberofHiddenNeurons
		 *            - 隐藏神经元个数
		 * @param ActivationFunction
		 *            - 激活函数 sig sin 两个可使用
		 * @param randomSeed
		 *            - 随机种子，为了算法可重现使用
		 * @param classIndex
		 *            - 预测或分类 属性标号，位置
		 */
		public Selm(int elm_type, int numberofHiddenNeurons,
				String ActivationFunction, int randomSeed, int classIndex) {
			Selm.m_seed = randomSeed;
			Selm.m_classAtt = classIndex;

			Elm_Type = elm_type;
			NumberofHiddenNeurons = numberofHiddenNeurons;
			func = ActivationFunction;
			NumberofOutputNeurons = 1; // 默认为一，数值型输出

			TrainingTime = 0;
			TestingTime = 0;
			TrainingAccuracy = 0;
			TestingAccuracy = 0;
		}

		public Selm() {

		}

		// by myself
		public static DenseMatrix TransMatrixAtt(DenseMatrix source, int att) {
			double temp;
			int rows = source.numRows();
			for (int i = 0; i < rows; i++) {
				temp = source.get(i, 0);
				source.set(i, 0, source.get(i, att));
				source.set(i, att, temp);
			}
			return source;
		}

		// --------3
		// by myself
		public static double[][] TransMatrixAtt(double[][] source, int rows,
				int att) {
			double temp;
			for (int i = 0; i < rows; i++) {
				temp = source[i][0];
				source[i][0] = source[i][att];
				source[i][att] = temp;
			}
			return source;
		}

		// --------4
		/**
		 * 如果为名词性预测，提取 预测的名词 个数 ，从 0 开始 ，所以要 +1
		 * 
		 * @param traindata
		 * @throws NotConvergedException
		 */
		public static void train(double[][] traindata)
				throws NotConvergedException {

			// classification require a the number of class

			train_set = new DenseMatrix(traindata);
			// train_set = TransMatrixAtt(train_set, m_classAtt);
			int m = train_set.numRows();
			if (Elm_Type == 1) {
				double maxtag = traindata[0][0];
				for (int i = 0; i < m; i++) {
					if (traindata[i][0] > maxtag)
						maxtag = traindata[i][0];
				}
				NumberofOutputNeurons = (int) maxtag + 1;
			}

			train();
		}

		/*
		 * public void WriteToFile(double[][] source,int r,int c, String
		 * fileName) { try { DecimalFormat format = new
		 * DecimalFormat("##0.00000000"); BufferedWriter writer = new
		 * BufferedWriter(new FileWriter(new File( fileName))); writer.write(r +
		 * " " + c); for (int i = 0; i < r; i++) { writer.newLine(); for (int j
		 * = 0; j < c; j++) {
		 * writer.write(String.valueOf(format.format(source[i][j])) + ' '); } }
		 * writer.flush(); writer.close();
		 * 
		 * } catch (IOException e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); } }
		 */

		// --------2
		// by myself
		/**
		 * 将数据归一化为双精度 二维数组
		 * 
		 * @param instances
		 * @param nomalization
		 * @throws NotConvergedException
		 * @throws NotSerializableException
		 */
		public void train(Instances instances, double[][] nomalization)
				throws NotConvergedException, NotSerializableException {
			int rows = instances.numInstances();
			int columns = instances.numAttributes();
			double[][] traindata = new double[rows][columns];

			/**
			 * 数值属性归一化 x-min/(max-min)
			 */
			for (int j = 0; j < columns; j++) {
				if (instances.attribute(j).isNumeric()) {
					for (int i = 0; i < rows; i++) {
						traindata[i][j] = instances.instance(i).value(j)
								- nomalization[1][j];
						traindata[i][j] /= nomalization[0][j]
								- nomalization[1][j];
					}
				} else {
					for (int i = 0; i < rows; i++) {
						traindata[i][j] = instances.instance(i).value(j);
					}
				}
			}

			/*
			 * for(int k=0; k<rows; k++){
			 * System.out.println(java.util.Arrays.toString(traindata[k])); }
			 */
			TransMatrixAtt(traindata, rows, m_classAtt);// ELM 算法
														// 将第一列作为预测属性，将原来的预测属性转至第一列
			/*
			 * for(int k=0; k<rows; k++){
			 * System.out.println(java.util.Arrays.toString(traindata[k])); }
			 */
			// WriteToFile(traindata, rows, columns, "D:\\weather.txt");

			train(traindata);
		}

		// --------5
		// by my self
		/**
		 * 随机化 矩阵 因为有随机种子，随机数可控，算法结果可重现
		 * 
		 * @param rows
		 * @param columns
		 * @param seed
		 * @return
		 */
		public static DenseMatrix randomMatrix(int rows, int columns, int seed) {
			Random x = new Random(seed);
			DenseMatrix source = new DenseMatrix(rows, columns);
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					source.set(i, j, x.nextDouble());
				}
			}
			return source;
		}

		// --------4
		/**
		 * 训练数据
		 * 
		 * @throws NotConvergedException
		 */
		private static void train() throws NotConvergedException {
			System.out.println("training");
			numTrainData = train_set.numRows();// 训练数据行数
			NumberofInputNeurons = train_set.numColumns() - 1;// 输入属性为全部属性减去输出属性
			/**
			 * 这样随机出的矩阵不可控，结果不可重现 InputWeight = (DenseMatrix)
			 * Matrices.random(NumberofHiddenNeurons, NumberofInputNeurons);
			 */
			InputWeight = randomMatrix(NumberofHiddenNeurons,
					NumberofInputNeurons, m_seed);

			DenseMatrix transT = new DenseMatrix(numTrainData, 1);// transT(numTrainData,1)
			DenseMatrix transP = new DenseMatrix(numTrainData,// transP(numTrainData,NumberofInputNeurons)
					NumberofInputNeurons);
			for (int i = 0; i < numTrainData; i++) {
				transT.set(i, 0, train_set.get(i, 0));
				for (int j = 1; j <= NumberofInputNeurons; j++)
					transP.set(i, j - 1, train_set.get(i, j));
			}
			// WriteToFile(InputWeight,"InputWeight.txt",1);

			T = new DenseMatrix(1, numTrainData);// T(1,numTrainData)
			DenseMatrix P = new DenseMatrix(NumberofInputNeurons, numTrainData);// P(NumberofInputNeurons,numTrainData)
			transT.transpose(T);// T = transT 转置
			transP.transpose(P);// P = transP 转置
			System.out.println(Elm_Type);
			if (Elm_Type != 0) // CLASSIFIER
			{
				label = new int[NumberofOutputNeurons];
				for (int i = 0; i < NumberofOutputNeurons; i++) {
					label[i] = i; // class label starts form 0
				}
				DenseMatrix tempT = new DenseMatrix(NumberofOutputNeurons,// tempT(NumberofOutputNeurons,numTrainData)
						numTrainData);
				tempT.zero();

				System.out.println(NumberofOutputNeurons);

				for (int i = 0; i < numTrainData; i++) {
					int j = 0;
					for (j = 0; j < NumberofOutputNeurons; j++) {
						if (label[j] == T.get(0, i))
							break;
					}
					tempT.set(j, i, 1);
				}

				T = new DenseMatrix(NumberofOutputNeurons, numTrainData); // T=temp_T*2-1;
				for (int i = 0; i < NumberofOutputNeurons; i++) {
					for (int j = 0; j < numTrainData; j++)
						T.set(i, j, tempT.get(i, j) * 2 - 1);
				}

				transT = new DenseMatrix(numTrainData, NumberofOutputNeurons);
				T.transpose(transT);

			} // end if CLASSIFIER

			long start_time_train = System.currentTimeMillis();
			// Random generate input weights InputWeight (w_i) and biases
			// BiasofHiddenNeurons (b_i) of hidden neurons
			// InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;

			/**
			 * BiasofHiddenNeurons = (DenseMatrix) Matrices.random(
			 * NumberofHiddenNeurons, 1);
			 */
			BiasofHiddenNeurons = randomMatrix(NumberofHiddenNeurons, 1, m_seed);

			DenseMatrix tempH = new DenseMatrix(NumberofHiddenNeurons,// tempH(NumberofHiddenNeurons,numTrainData)
					numTrainData);
			InputWeight.mult(P, tempH);

			// WriteToFile(tempH, "myOutput", 1);// write to file {the matrix}

			// tempH = InputWeight * P

			// DenseMatrix ind = new DenseMatrix(1, numTrainData);

			DenseMatrix BiasMatrix = new DenseMatrix(NumberofHiddenNeurons,// BiasMatrix(NumberofHiddenNeurons,numTrainData)
					numTrainData);

			for (int j = 0; j < numTrainData; j++) {
				for (int i = 0; i < NumberofHiddenNeurons; i++) {
					BiasMatrix.set(i, j, BiasofHiddenNeurons.get(i, 0));
				}
			}

			tempH.add(BiasMatrix);
			DenseMatrix H = new DenseMatrix(NumberofHiddenNeurons, numTrainData);

			if (func.startsWith("sig")) {
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTrainData; i++) {
						double temp = tempH.get(j, i);
						temp = 1.0f / (1 + Math.exp(-temp));
						H.set(j, i, temp);
					}
				}
			} else if (func.startsWith("sin")) {
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTrainData; i++) {
						double temp = tempH.get(j, i);
						temp = Math.sin(temp);
						H.set(j, i, temp);
					}
				}
			} else if (func.startsWith("hardlim")) {
				// If you need it ,you can absolutely complete it yourself
			} else if (func.startsWith("tribas")) {
				// If you need it ,you can absolutely complete it yourself
			} else if (func.startsWith("radbas")) {
				// If you need it ,you can absolutely complete it yourself
				double a = 2, b = 2, c = Math.sqrt(2);
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTrainData; i++) {
						double temp = tempH.get(j, i);
						temp = a * Math.exp(-(temp - b) * (temp - b) / c * c);
						H.set(j, i, temp);
					}
				}
			}

			DenseMatrix Ht = new DenseMatrix(numTrainData,// Ht(numTrainData,NumberofHiddenNeurons)
					NumberofHiddenNeurons);
			H.transpose(Ht);

			Inverse invers = new Inverse(Ht);
			System.out.println("Ht Inverse...");
			DenseMatrix pinvHt = invers.getMPInverse(); // NumberofHiddenNeurons*numTrainData
			// DenseMatrix pinvHt = invers.getMPInverse(0.000001); //fast
			// method,
			// PLEASE CITE in your paper properly:
			// Guang-Bin Huang, Hongming Zhou, Xiaojian Ding, and Rui Zhang,
			// "Extreme Learning Machine for Regression and Multi-Class Classification,"
			// submitted to IEEE Transactions on Pattern Analysis and Machine
			// Intelligence, October 2010.

			OutputWeight = new DenseMatrix(NumberofHiddenNeurons,// outputWeight(NumberofHiddenNeurons,NumberofOutputNeurons)
					NumberofOutputNeurons);
			pinvHt.mult(transT, OutputWeight);// OutputWeight=pinv(H') * T';

			long end_time_train = System.currentTimeMillis();
			TrainingTime = (end_time_train - start_time_train) * 1.0f / 1000;

			DenseMatrix Yt = new DenseMatrix(numTrainData,// Yt(numTrainData,NumberofOutputNeurons)
					NumberofOutputNeurons);
			Ht.mult(OutputWeight, Yt);
			Y = new DenseMatrix(NumberofOutputNeurons, numTrainData);// Y(NumberofOutputNeurons,numTrainData)

			Yt.transpose(Y);
			if (Elm_Type == 0) {
				double MSE = 0;
				for (int i = 0; i < numTrainData; i++) {
					MSE += (Yt.get(i, 0) - transT.get(i, 0))
							* (Yt.get(i, 0) - transT.get(i, 0));
				}
				TrainingAccuracy = Math.sqrt(MSE / numTrainData);
			}

			// CLASSIFIER
			else if (Elm_Type == 1) {
				float MissClassificationRate_Training = 0;

				for (int i = 0; i < numTrainData; i++) {
					double maxtag1 = Y.get(0, i);
					int tag1 = 0;
					double maxtag2 = T.get(0, i);
					int tag2 = 0;
					for (int j = 1; j < NumberofOutputNeurons; j++) {
						if (Y.get(j, i) > maxtag1) {
							maxtag1 = Y.get(j, i);
							tag1 = j;
						}
						if (T.get(j, i) > maxtag2) {
							maxtag2 = T.get(j, i);
							tag2 = j;
						}
					}
					if (tag1 != tag2)
						MissClassificationRate_Training++;
				}
				TrainingAccuracy = 1 - MissClassificationRate_Training * 1.0f
						/ numTrainData;

			}
			System.out.println("calculate ....");// ..........................
		}

		public double[] testOut(double[][] inpt, int r, int c) {
			test_set = new DenseMatrix(inpt);
			return testOut(r, c);
		}

		/*
		 * public double[] testOut(double[] inpt, int r, int c) { test_set = new
		 * DenseMatrix(new DenseVector(inpt)); return testOut(r, c); }
		 */

		// Output numTestData*NumberofOutputNeurons
		private double[] testOut(int rows, int columns) {
			numTestData = rows;// test_set.numRows();
			NumberofInputNeurons = columns;// test_set.numColumns();// 问题
			// System.out.println(numTestData+","+NumberofInputNeurons+","+test_set.numColumns());
			DenseMatrix ttestT = new DenseMatrix(numTestData, 1);
			DenseMatrix ttestP = new DenseMatrix(numTestData,
					NumberofInputNeurons);
			for (int i = 0; i < numTestData; i++) {
				ttestT.set(i, 0, test_set.get(i, 0));
				for (int j = 0; j < NumberofInputNeurons; j++)
					ttestP.set(i, j, test_set.get(i, j));
			}

			testT = new DenseMatrix(1, numTestData);
			testP = new DenseMatrix(NumberofInputNeurons, numTestData);
			ttestT.transpose(testT);
			ttestP.transpose(testP);
			// test_set.transpose(testP);
			// System.out.println(NumberofHiddenNeurons+" "+numTestData);
			DenseMatrix tempH_test = new DenseMatrix(NumberofHiddenNeurons,
					numTestData);
			// System.out.println(InputWeight.numRows()+" "+InputWeight.numColumns());
			// System.out.println(testP.numRows()+" "+testP.numColumns());
			InputWeight.mult(testP, tempH_test);
			DenseMatrix BiasMatrix2 = new DenseMatrix(NumberofHiddenNeurons,
					numTestData);
			for (int j = 0; j < numTestData; j++) {
				for (int i = 0; i < NumberofHiddenNeurons; i++) {
					BiasMatrix2.set(i, j, BiasofHiddenNeurons.get(i, 0));
				}
			}

			tempH_test.add(BiasMatrix2);
			DenseMatrix H_test = new DenseMatrix(NumberofHiddenNeurons,
					numTestData);

			if (func.startsWith("sig")) {
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTestData; i++) {
						double temp = tempH_test.get(j, i);
						temp = 1.0f / (1 + Math.exp(-temp));
						H_test.set(j, i, temp);
					}
				}
			} else if (func.startsWith("sin")) {
				for (int j = 0; j < NumberofHiddenNeurons; j++) {
					for (int i = 0; i < numTestData; i++) {
						double temp = tempH_test.get(j, i);
						temp = Math.sin(temp);
						H_test.set(j, i, temp);
					}
				}
			} else if (func.startsWith("hardlim")) {

			} else if (func.startsWith("tribas")) {

			} else if (func.startsWith("radbas")) {

			}

			DenseMatrix transH_test = new DenseMatrix(numTestData,
					NumberofHiddenNeurons);
			H_test.transpose(transH_test);
			DenseMatrix Yout = new DenseMatrix(numTestData,
					NumberofOutputNeurons);
			transH_test.mult(OutputWeight, Yout);

			// DenseMatrix testY = new
			// DenseMatrix(NumberofOutputNeurons,numTestData);
			// Yout.transpose(testY);

			double[] result = new double[numTestData];

			if (Elm_Type == 0) {
				for (int i = 0; i < numTestData; i++)
					result[i] = Yout.get(i, 0);
			}

			else if (Elm_Type == 1) {
				for (int i = 0; i < numTestData; i++) {
					int tagmax = 0;
					double tagvalue = Yout.get(i, 0);
					for (int j = 1; j < NumberofOutputNeurons; j++) {
						if (Yout.get(i, j) > tagvalue) {
							tagvalue = Yout.get(i, j);
							tagmax = j;
						}

					}
					result[i] = tagmax;
				}
			}
			return result;
		}

		public float getTrainingTime() {
			return TrainingTime;
		}

		public double getTrainingAccuracy() {
			return TrainingAccuracy;
		}

		public float getTestingTime() {
			return TestingTime;
		}

		public double getTestingAccuracy() {
			return TestingAccuracy;
		}

		public int getNumberofInputNeurons() {
			return NumberofInputNeurons;
		}

		public int getNumberofHiddenNeurons() {
			return NumberofHiddenNeurons;
		}

		public int getNumberofOutputNeurons() {
			return NumberofOutputNeurons;
		}

		public DenseMatrix getInputWeight() {
			return InputWeight;
		}

		public DenseMatrix getBiasofHiddenNeurons() {
			return BiasofHiddenNeurons;
		}

		public DenseMatrix getOutputWeight() {
			return OutputWeight;
		}

		public DenseMatrix getTrainSet() {
			return Selm.train_set;
		}
	}

	/* 随机种子 用于初始化矩阵 */
	private int m_randomSeed = 1;
	/* 激活函数 */
	private String m_activeFunction = "sig";
	/* 隐藏层 神经元个数 */
	private int m_numberofHiddenNeurons = 20;
	/**/
	private Selm m_elm = null;
	/**
	 * 归一化数组，标准化 数值型属性，取最大值 Max 减去最小值 Min 为除数，原值 x 减最小值 Min 为被除数
	 * 数组取两行，第一行为最大值，第二行为最小值
	 */
	private double[][] m_nomalization;

	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// TODO Auto-generated method stub

		// remove instances with missing class
		Instances data = new Instances(instances);
		data.deleteWithMissingClass();

		int length = instances.numAttributes();
		m_nomalization = new double[2][length];
		for (int i = 0; i < length; i++) {
			if (instances.attribute(i).isNumeric()) {
				m_nomalization[0][i] = instances.attributeStats(i).numericStats.max;
				m_nomalization[1][i] = instances.attributeStats(i).numericStats.min;
			}
		}
		// System.out.println(java.util.Arrays.toString(m_nomalization[0]));
		// System.out.println(java.util.Arrays.toString(m_nomalization[1]) +
		// "\n");

		int elm_type = 0;
		if (instances.classAttribute().isNominal()) {
			elm_type = 1;
		}
		int classIndex = instances.classIndex();
		m_elm = new Selm(elm_type, m_numberofHiddenNeurons, m_activeFunction,
				m_randomSeed, classIndex);
		m_elm.train(instances, m_nomalization);
		// System.out.println("training over" +
		// elm_type+m_numberofHiddenNeurons+m_function+m_randomSeed+classIndex);
	}

	/**
	 * Classifies a given instance.
	 * 
	 * @param instance
	 *            the instance to be classified
	 * @return index of the predicted class
	 */
	@Override
	public double classifyInstance(Instance instance) {
		for (int j = 0; j < instance.numAttributes(); j++) {
			if (instance.attribute(j).isNumeric()) {
				instance.setValue(j, (instance.value(j) - m_nomalization[1][j])
						/ (m_nomalization[0][j] - m_nomalization[1][j]));
			}
		}
		instance.setValue(instance.classIndex(), instance.value(0));
		int columns = instance.numAttributes() - 1;
		double[][] predict = new double[1][columns];
		for (int i = 1; i < instance.numAttributes(); i++) {
			predict[0][i - 1] = instance.value(i);
		}
		// System.out.println(instance.numAttributes() +" -- "
		// +instance.classIndex());
		// System.out.println(java.util.Arrays.toString(predict[0]));
		double[] result = m_elm.testOut(predict, 1, columns);
		//System.out.println(java.util.Arrays.toString(result));
		if (instance.attribute(instance.classIndex()).isNominal()) {
			return result[0];
		}
		return result[0]* (m_nomalization[0][instance.classIndex()] - 
				m_nomalization[1][instance.classIndex()])+ m_nomalization[1][instance.classIndex()];
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {
		return "Extreme Learning Mechine";
	}

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for building and using a ELM classifier;"
				+ " this code modified by Ye Qiangsheng, the source code author is DongLi (follow website) \n\n"
				+ getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Author: Guanbinhuang");
	    result.setValue(Field.YEAR, "Year: 2004");
	    result.setValue(Field.TITLE,
	        " http://www.ntu.edu.sg/home/egbhuang/elm_codes.html ");
		return result;
	}

	 /**
	   * Returns default capabilities of the classifier.
	   * 
	   * @return the capabilities of this classifier
	   */
	  @Override
	  public Capabilities getCapabilities() {
	    Capabilities result = super.getCapabilities();
	    result.disableAll();

	    // attributes
	    result.enable(Capability.NOMINAL_ATTRIBUTES);
	    result.enable(Capability.NUMERIC_ATTRIBUTES);
	   // result.enable(Capability.DATE_ATTRIBUTES);
	    result.enable(Capability.STRING_ATTRIBUTES);
	    result.enable(Capability.RELATIONAL_ATTRIBUTES);
	    //result.enable(Capability.MISSING_VALUES);

	    // class
	    result.enable(Capability.NOMINAL_CLASS);
	    result.enable(Capability.NUMERIC_CLASS);
	   // result.enable(Capability.DATE_CLASS);
	    //result.enable(Capability.MISSING_CLASS_VALUES);

	    // instances
	    result.setMinimumNumberInstances(0);

	    return result;
	  }
	/**
	 * Returns an enumeration describing the available options..
	 * 
	 * @return an enumeration of all the available options.
	 */
	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(14);

		newVector.addElement(new Option(
				"\t number of HiddenNeurons , default 20. \n"
				+ "\tbigger than 0, more bigger more better but slowly.",
				"N", 1, "-N number of HiddenNeurons"));
		newVector.addElement(new Option(
				"\t random seed ,default 1\n",
				"R", 1, "-R random seed "));
		newVector.addElement(new Option(
				"\t active function \n", "F", 1, "-N active function"));

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * 
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		String numberofHiddenNeuronsString = Utils.getOption('N', options);
		if (numberofHiddenNeuronsString.length() != 0) {
			m_numberofHiddenNeurons = Integer
					.parseInt(numberofHiddenNeuronsString);
		} else {
			m_numberofHiddenNeurons = 20;
		}

		String randomSeedString = Utils.getOption('R', options);
		if (randomSeedString.length() != 0) {
			m_randomSeed = Integer.parseInt(randomSeedString);
		} else {
			m_randomSeed = 1;
		}

		String activeFunctionString = Utils.getOption('F', options);
		if (activeFunctionString.length() != 0) {
			m_activeFunction = activeFunctionString;
		} else {
			m_activeFunction = "sig";
		}

		super.setOptions(options);
	}

	/**
	 * 
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>(3);

		options.add("-N");
		options.add("" + m_numberofHiddenNeurons);
		options.add("-R");
		options.add("" + m_randomSeed);
		options.add("-F");
		options.add("" + m_activeFunction);

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}
	  
	/**
	 * Main method for testing this class.
	 * 
	 * @param argv
	 *            the options
	 */
	public static void main(String[] argv) {
		runClassifier(new ELM(), argv);
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String randomSeedTipText() {
		return "the random seed to initial the matrix";
	}
	
	public int getRandomSeed() {
		return m_randomSeed;
	}

	public void setRandomSeed(int m_randomSeed) {
		this.m_randomSeed = m_randomSeed;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String activeFunctionTipText() {
		return "the active function";
	}
	
	public String getActiveFunction() {
		return m_activeFunction;
	}

	public void setActiveFunction(String m_function) {
		this.m_activeFunction = m_function;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String numberofHiddenNeuronsTipText() {
		return "number of hidden neurons ";
	}
	
	public int getNumberofHiddenNeurons() {
		return m_numberofHiddenNeurons;
	}

	public void setNumberofHiddenNeurons(int m_numberofHiddenNeurons) {
		this.m_numberofHiddenNeurons = m_numberofHiddenNeurons;
	}

}
