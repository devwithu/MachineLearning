#include "../dlib-18.6/dlib/svm.h"
#include <iostream>
#include <conio.h>

using namespace dlib;
using namespace std;

/*

	traning set
	------------------------------------
	TEMPerature   RAIN   WIND    | play
	------------------------------------
	5               NO    YES    |   NO
	10              NO     NO    |  YES
	8               NO     NO    |   NO
	6              YES     NO    |   NO
	24              NO    YES    |  YES
	30              NO     NO    |  YES
	33              NO     NO    |   NO
	32              NO     NO    |   NO
	30             YES     NO    |   NO
	15             YES     NO    |   NO
	16              NO    YES    |  YES

*/

/* field */
#define TEMP	0
#define RAIN	1
#define WIND	2

/* value */
#define YES		(1)
#define NO		(-1)

// 0���� ���ų� ũ��, "YES", ������ "NO" ����
LPCSTR Say(IN double fResult) { if (fResult >= 0) return "YES"; return "NO"; }

void main()
{
	unsigned long i = 0;

	// Attribute�� 3���̴�. (�µ�, ��, �ٶ�)
	// matrix�� ǥ���Ѵ�.
	typedef matrix<double, 3, 1> CTrainingNode;

	// non-linear SVM
	// �Ϲ������� ��-���� ���� ���ȴ�.
    typedef radial_basis_kernel<CTrainingNode> ST_NONLINEAR_KERNEL;

	// Training Set�� Node�� �ϳ��ϳ� �߰��ϱ� ���� �ʿ��ϴ�.
	CTrainingNode cTrainingNode;

	// Training Node�� Vector ���·� �̷��� Training Set�̴�.
	std::vector<CTrainingNode> cArrTrainingSet;

	// training set���� Play�� Yes/No ���θ� ������ Label
	std::vector<double> cArrLabelPlay;

	// ������ Node�� �߰��Ѵ�.
	cTrainingNode(TEMP) = 5;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = YES;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 10;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(YES);

	cTrainingNode(TEMP) = 8;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 6;
	cTrainingNode(RAIN) = YES;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 24;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = YES;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(YES);

	cTrainingNode(TEMP) = 30;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(YES);

	cTrainingNode(TEMP) = 33;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 32;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 30;
	cTrainingNode(RAIN) = YES;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 15;
	cTrainingNode(RAIN) = YES;
	cTrainingNode(WIND) = NO;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(NO);

	cTrainingNode(TEMP) = 16;
	cTrainingNode(RAIN) = NO;
	cTrainingNode(WIND) = YES;
	cArrTrainingSet.push_back(cTrainingNode);
	cArrLabelPlay.push_back(YES);

	// Training set�� normalize�ϱ� ���� class instance�� �����Ѵ�.
	vector_normalizer<CTrainingNode> cNormalizer;

	// Training set�� normalize �Ѵ�. ���� ���� �ʾƵ� �ȴ�.
	// �ʿ����� �ʴٸ�, cNormalizer�� ���õ� ��� line�� ��������.
	cNormalizer.train(cArrTrainingSet);

	// ������ ���� ������ �������, Training set�� ��� element���� normalize�Ѵ�.
	// ���� ���,
	// ù��° node��,
	//	cTrainingNode(TEMP) = 5;
	//	cTrainingNode(RAIN) = -1; // NO
	//	cTrainingNode(WIND) = 1;  // YES
	// ��,
	//	cTrainingNode(TEMP) = -1.2654276706088274;
	//	cTrainingNode(RAIN) = -0.58387420812114232;
	//	cTrainingNode(WIND) = 1.5569978883230462;
	// �� ��ȯ�ȴ�.
	for (i=0; i<cArrTrainingSet.size(); i++)
	{
		cArrTrainingSet[i] = cNormalizer(cArrTrainingSet[i]);
	}

	// Trainer�� �����Ѵ�.
	svm_nu_trainer<ST_NONLINEAR_KERNEL> cTrainer;

	// Trainer�� kernel ���� �����Ѵ�.
    // cTrainer.set_kernel(ST_NONLINEAR_KERNEL(0.15625));
    // cTrainer.set_nu(0.15625);

	// �н� ��� �Լ�(���� �Լ�, decision function)�� �����Ѵ�.
	// 0���� ���� ���� -1,
	// 0���� ���ų� ū ���� +1
	// �� �з��ȴ�.
	typedef decision_function<ST_NONLINEAR_KERNEL> ST_TYPE_FUNCTION;

	// �н��� normalize ������ �̿��ϵ��� �Ѵ�.
	typedef normalized_function<ST_TYPE_FUNCTION> ST_TYPE_NORMALIZE_FUNCTION;

	// Normalize�� �⺻������ ��� �Լ� instance�� �����Ѵ�.
	ST_TYPE_NORMALIZE_FUNCTION stFunction;

	// Normalize ������ �����Ѵ�.
	stFunction.normalizer = cNormalizer;

	// �н��� �����Ѵ� !!!
	stFunction.function = cTrainer.train(cArrTrainingSet, cArrLabelPlay);

	// ����, ���, ������ ���е� ���� �ƴ�,
	// 0~1 ������ Ȯ���� �����ϴ� ��� �Լ��� ����� ���� �н��� ����.
	typedef probabilistic_decision_function<ST_NONLINEAR_KERNEL> ST_TYPE_FUNCTION_PROB;

	// ����, �̰͵� �н��� normalize ������ �̿��Ѵ�.
	typedef normalized_function<ST_TYPE_FUNCTION_PROB> ST_TYPE_NORMALIZE_FUNCTION_PROB;

	// Normalize�� �⺻���� �� ��� �Լ� instance�� �����Ѵ�. 0~1 ������ Ȯ������ �����Ѵ�.
	ST_TYPE_NORMALIZE_FUNCTION_PROB stFunctionProb;

	// Normalize ������ �����Ѵ�.
	stFunctionProb.normalizer = cNormalizer;

	// �н��� �����Ѵ� !!!
	stFunctionProb.function = train_probabilistic_decision_function(cTrainer, cArrTrainingSet, cArrLabelPlay, 3);

	/*
	------------------------------------
	temperature   rain   wind    | play
	------------------------------------
	10              NO     NO    |    ?
	11             YES    YES    |    ?
	15              NO     NO    |    ?
	33              NO     NO    |    ?
	32              NO     NO    |    ?
	31              NO     NO    |    ?
	10             YES     NO    |    ?
	*/

	CTrainingNode question;

	question(TEMP) = 10;
	question(RAIN) = NO;
	question(WIND) = NO;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", 
		(int)question(TEMP), 
		Say(question(RAIN)),
		Say(question(WIND)),
		Say(stFunction(question)),
		stFunction(question),			// �н� ����� ����. 0 ���� �̻� : +, 0 �̸� : -
		stFunctionProb(question));		// �н� ����� ����. 0 ~ 1 ����, +�� Ȯ��

	question(TEMP) = 11;
	question(RAIN) = YES;
	question(WIND) = YES;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", (int)question(TEMP), Say(question(RAIN)), Say(question(WIND)), Say(stFunction(question)), stFunction(question), stFunctionProb(question));

	question(TEMP) = 15;
	question(RAIN) = NO;
	question(WIND) = NO;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", (int)question(TEMP), Say(question(RAIN)), Say(question(WIND)), Say(stFunction(question)), stFunction(question), stFunctionProb(question));

	question(TEMP) = 33;
	question(RAIN) = NO;
	question(WIND) = NO;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", (int)question(TEMP), Say(question(RAIN)), Say(question(WIND)), Say(stFunction(question)), stFunction(question), stFunctionProb(question));

	question(TEMP) = 32;
	question(RAIN) = NO;
	question(WIND) = NO;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", (int)question(TEMP), Say(question(RAIN)), Say(question(WIND)), Say(stFunction(question)), stFunction(question), stFunctionProb(question));

	question(TEMP) = 31;
	question(RAIN) = NO;
	question(WIND) = NO;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", (int)question(TEMP), Say(question(RAIN)), Say(question(WIND)), Say(stFunction(question)), stFunction(question), stFunctionProb(question));

	question(TEMP) = 10;
	question(RAIN) = YES;
	question(WIND) = NO;
	printf("temperature: %d, rain=%s, win=%s, output=%s(%f), probability=%f\r\n", (int)question(TEMP), Say(question(RAIN)), Say(question(WIND)), Say(stFunction(question)), stFunction(question), stFunctionProb(question));

	printf("\r\n");
	printf("10-fold cross validation accuracy : %f\r\n", cross_validate_trainer(reduced2(cTrainer,10), cArrTrainingSet, cArrLabelPlay, 3));

	printf("Press a key to exit.\r\n");
	getch();
	return;
}