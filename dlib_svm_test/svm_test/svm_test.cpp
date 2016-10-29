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

// 0보다 같거나 크면, "YES", 작으면 "NO" 리턴
LPCSTR Say(IN double fResult) { if (fResult >= 0) return "YES"; return "NO"; }

void main()
{
	unsigned long i = 0;

	// Attribute가 3개이다. (온도, 비, 바람)
	// matrix로 표현한다.
	typedef matrix<double, 3, 1> CTrainingNode;

	// non-linear SVM
	// 일반적으로 비-선형 모델이 사용된다.
    typedef radial_basis_kernel<CTrainingNode> ST_NONLINEAR_KERNEL;

	// Training Set에 Node를 하나하나 추가하기 위해 필요하다.
	CTrainingNode cTrainingNode;

	// Training Node의 Vector 형태로 이뤄진 Training Set이다.
	std::vector<CTrainingNode> cArrTrainingSet;

	// training set에서 Play의 Yes/No 여부를 가지는 Label
	std::vector<double> cArrLabelPlay;

	// 각각의 Node를 추가한다.
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

	// Training set을 normalize하기 위한 class instance를 생성한다.
	vector_normalizer<CTrainingNode> cNormalizer;

	// Training set을 normalize 한다. 굳이 하지 않아도 된다.
	// 필요하지 않다면, cNormalizer에 관련된 모든 line을 삭제하자.
	cNormalizer.train(cArrTrainingSet);

	// 위에서 계산된 정보를 기반으로, Training set의 모든 element들을 normalize한다.
	// 예를 들어,
	// 첫번째 node인,
	//	cTrainingNode(TEMP) = 5;
	//	cTrainingNode(RAIN) = -1; // NO
	//	cTrainingNode(WIND) = 1;  // YES
	// 는,
	//	cTrainingNode(TEMP) = -1.2654276706088274;
	//	cTrainingNode(RAIN) = -0.58387420812114232;
	//	cTrainingNode(WIND) = 1.5569978883230462;
	// 로 변환된다.
	for (i=0; i<cArrTrainingSet.size(); i++)
	{
		cArrTrainingSet[i] = cNormalizer(cArrTrainingSet[i]);
	}

	// Trainer를 생성한다.
	svm_nu_trainer<ST_NONLINEAR_KERNEL> cTrainer;

	// Trainer의 kernel 값을 전달한다.
    // cTrainer.set_kernel(ST_NONLINEAR_KERNEL(0.15625));
    // cTrainer.set_nu(0.15625);

	// 학습 결과 함수(결정 함수, decision function)를 정의한다.
	// 0보다 작은 값은 -1,
	// 0보다 같거나 큰 값은 +1
	// 로 분류된다.
	typedef decision_function<ST_NONLINEAR_KERNEL> ST_TYPE_FUNCTION;

	// 학습시 normalize 정보를 이용하도록 한다.
	typedef normalized_function<ST_TYPE_FUNCTION> ST_TYPE_NORMALIZE_FUNCTION;

	// Normalize를 기본으로한 결과 함수 instance를 생성한다.
	ST_TYPE_NORMALIZE_FUNCTION stFunction;

	// Normalize 정보를 전달한다.
	stFunction.normalizer = cNormalizer;

	// 학습을 시작한다 !!!
	stFunction.function = cTrainer.train(cArrTrainingSet, cArrLabelPlay);

	// 이제, 양수, 음수로 구분된 값이 아닌,
	// 0~1 사이의 확률을 전달하는 결과 함수를 만들어 보고 학습해 보자.
	typedef probabilistic_decision_function<ST_NONLINEAR_KERNEL> ST_TYPE_FUNCTION_PROB;

	// 역시, 이것도 학습시 normalize 정보를 이용한다.
	typedef normalized_function<ST_TYPE_FUNCTION_PROB> ST_TYPE_NORMALIZE_FUNCTION_PROB;

	// Normalize를 기본으로 한 결과 함수 instance를 생성한다. 0~1 사이의 확률값을 전달한다.
	ST_TYPE_NORMALIZE_FUNCTION_PROB stFunctionProb;

	// Normalize 정보를 전달한다.
	stFunctionProb.normalizer = cNormalizer;

	// 학습을 시작한다 !!!
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
		stFunction(question),			// 학습 결과값 전달. 0 포함 이상 : +, 0 미만 : -
		stFunctionProb(question));		// 학습 결과값 전달. 0 ~ 1 사이, +의 확율

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