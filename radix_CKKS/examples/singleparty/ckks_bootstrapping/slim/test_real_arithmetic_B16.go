package main

import (
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"runtime"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v6/circuits/ckks/bootstrapping"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/dft"
	"github.com/tuneinsight/lattigo/v6/circuits/ckks/mod1"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/ring"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func SampleFixedPrecision(k uint) *big.Float {
	// 2^k
	modulus := new(big.Int).Lsh(big.NewInt(1), k)

	// n uniformly sampled from {0, ..., 2^(k+1) - 1}
	rangeSize := new(big.Int).Lsh(big.NewInt(1), k+1)

	n, err := rand.Int(rand.Reader, rangeSize)
	if err != nil {
		panic(err)
	}

	// n <- n - 2^k
	// 이제 n ∈ {-2^k, ..., 2^k - 1}
	n.Sub(n, modulus)

	// x = n / 2^k
	// 따라서 x ∈ [-1, 1)
	r := new(big.Rat).SetFrac(n, modulus)

	return new(big.Float).
		SetPrec(k + 1).
		SetRat(r)
}

func SampleFixedPrecisionHalf(k uint) *big.Float {
	// 2^k
	modulus := new(big.Int).Lsh(big.NewInt(1), k)

	// n uniformly sampled from {0, ..., 2^k - 1}
	n, err := rand.Int(rand.Reader, modulus)
	if err != nil {
		panic(err)
	}

	// n <- n - 2^(k-1)
	// n ∈ {-2^(k-1), ..., 2^(k-1)-1}
	half := new(big.Int).Lsh(big.NewInt(1), k-1)
	n.Sub(n, half)

	// x = n / 2^k
	// x ∈ [-1/2, 1/2)
	r := new(big.Rat).SetFrac(n, modulus)

	return new(big.Float).
		SetPrec(k + 1).
		SetRat(r)
}

func MulAndRound(x *big.Float, n *big.Int) *big.Int {
	prec := x.Prec()

	// n -> big.Float
	nFloat := new(big.Float).
		SetPrec(prec).
		SetInt(n)

	// y = x * n
	y := new(big.Float).
		SetPrec(prec).
		Mul(x, nFloat)

	// ±0.5를 더해서 nearest integer rounding
	half := new(big.Float).
		SetPrec(prec).
		SetFloat64(0.5)

	if y.Sign() >= 0 {
		y.Add(y, half)
	} else {
		y.Sub(y, half)
	}

	// Int()는 0 방향으로 truncate
	result, _ := y.Int(nil)

	return result
}

func DecomposeBaseFixed(x *big.Int, B *big.Int, length int) []int {
	if B.Cmp(big.NewInt(1)) <= 0 {
		panic("base B must be greater than 1")
	}

	if x.Sign() < 0 {
		panic("x must be non-negative")
	}

	if length <= 0 {
		panic("length must be positive")
	}

	n := new(big.Int).Set(x)

	// 처음부터 고정 길이 + 0으로 초기화
	digits := make([]int, length)

	q := new(big.Int)
	r := new(big.Int)

	for i := 0; i < length && n.Sign() > 0; i++ {
		q.QuoRem(n, B, r)

		// LSB first
		digits[i] = int(r.Int64())

		n.Set(q)
	}

	// length로 표현할 수 없는 경우
	if n.Sign() > 0 {
		panic("x is too large for the given decomposition length")
	}

	return digits
}

func IntDivScale(x, scale *big.Int) *big.Float {
	if scale.Sign() == 0 {
		panic("scale must be non-zero")
	}

	prec := uint(max(x.BitLen(), scale.BitLen()) + 2000)

	xf := new(big.Float).
		SetPrec(prec).
		SetInt(x)

	sf := new(big.Float).
		SetPrec(prec).
		SetInt(scale)

	return new(big.Float).
		SetPrec(prec).
		Quo(xf, sf)
}

func ComposeBase(digits []int, B *big.Int) *big.Int {
	if B.Cmp(big.NewInt(1)) <= 0 {
		panic("base B must be greater than 1")
	}

	result := big.NewInt(0)
	power := big.NewInt(1)

	for _, d := range digits {
		if d < 0 {
			panic("digits must be non-negative")
		}

		if big.NewInt(int64(d)).Cmp(B) >= 0 {
			panic("digit must be smaller than base B")
		}

		term := new(big.Int).Mul(
			big.NewInt(int64(d)),
			power,
		)

		result.Add(result, term)

		// power *= B
		power.Mul(power, B)
	}

	return result
}

func Log2Error(a, b *big.Float, prec uint) *big.Float {
	// diff = |a - b|
	diff := new(big.Float).SetPrec(prec).Sub(a, b)
	diff.Abs(diff)

	if diff.Sign() == 0 {
		//panic("a and b are identical: log error is +Inf")
	}

	// diff = m * 2^exp, 0.5 <= m < 1
	m := new(big.Float).SetPrec(prec)
	exp := diff.MantExp(m)

	// -log2(diff) = -exp - log2(m)
	log2m := log2BigFloat(m, prec)

	result := new(big.Float).SetPrec(prec).SetInt64(int64(-exp))
	result.Sub(result, log2m)

	return result
}

func log2BigFloat(x *big.Float, prec uint) *big.Float {
	// ln(x) / ln(2)
	lnX := lnBigFloat(x, prec)

	two := new(big.Float).SetPrec(prec).SetInt64(2)
	ln2 := lnBigFloat(two, prec)

	return new(big.Float).SetPrec(prec).Quo(lnX, ln2)
}

func lnBigFloat(x *big.Float, prec uint) *big.Float {
	if x.Sign() <= 0 {
		panic("ln input must be positive")
	}

	// Normalize x = m * 2^e, 0.5 <= m < 1
	m := new(big.Float).SetPrec(prec)
	e := x.MantExp(m)

	one := new(big.Float).SetPrec(prec).SetInt64(1)

	// z = (m - 1) / (m + 1)
	num := new(big.Float).SetPrec(prec).Sub(m, one)
	den := new(big.Float).SetPrec(prec).Add(m, one)
	z := new(big.Float).SetPrec(prec).Quo(num, den)

	z2 := new(big.Float).SetPrec(prec).Mul(z, z)
	term := new(big.Float).SetPrec(prec).Set(z)
	sum := new(big.Float).SetPrec(prec).Set(z)

	// ln(m) = 2 * (z + z^3/3 + z^5/5 + ...)
	iterations := int(prec/2) + 32

	for n := 1; n < iterations; n++ {
		term.Mul(term, z2)

		divisor := new(big.Float).
			SetPrec(prec).
			SetInt64(int64(2*n + 1))

		tmp := new(big.Float).
			SetPrec(prec).
			Quo(term, divisor)

		sum.Add(sum, tmp)
	}

	sum.Mul(sum, new(big.Float).SetPrec(prec).SetInt64(2))

	// ln(x) = ln(m) + e * ln(2)
	//
	// ln(2)는 동일한 series를 0.5에 적용:
	// ln(0.5) = -ln(2)
	half := new(big.Float).SetPrec(prec).SetFloat64(0.5)

	num2 := new(big.Float).SetPrec(prec).Sub(half, one)
	den2 := new(big.Float).SetPrec(prec).Add(half, one)
	zL2 := new(big.Float).SetPrec(prec).Quo(num2, den2)

	zL22 := new(big.Float).SetPrec(prec).Mul(zL2, zL2)
	term2 := new(big.Float).SetPrec(prec).Set(zL2)
	sum2 := new(big.Float).SetPrec(prec).Set(zL2)

	for n := 1; n < iterations; n++ {
		term2.Mul(term2, zL22)

		divisor := new(big.Float).
			SetPrec(prec).
			SetInt64(int64(2*n + 1))

		tmp := new(big.Float).
			SetPrec(prec).
			Quo(term2, divisor)

		sum2.Add(sum2, tmp)
	}

	sum2.Mul(sum2, new(big.Float).SetPrec(prec).SetInt64(2))

	// ln(2) = -ln(0.5)
	ln2 := new(big.Float).SetPrec(prec).Neg(sum2)

	eFloat := new(big.Float).
		SetPrec(prec).
		SetInt64(int64(e))

	eLn2 := new(big.Float).
		SetPrec(prec).
		Mul(eFloat, ln2)

	return new(big.Float).
		SetPrec(prec).
		Add(sum, eLn2)
}

func Test_real_arthmetic_B16(radix_params RParams, setting bool) {

	//==============================================
	//=== 0) Set Large CKKS paramter  ==============
	//==============================================

	// We use Single Thread !!
	runtime.GOMAXPROCS(1)

	fmt.Println()
	fmt.Println("=== Scheme Parameter ===")

	// Default LogN, which with the following defined parameters
	// provides a security of 128-bit.
	LogN := 16

	bit_length := radix_params.bit_length
	base := radix_params.B
	//base_bit := Bbox.base_bit
	slice_length := radix_params.slice_length
	//log_slice_half := Bbox.log_slice_half

	var LogDefaultScale int

	var Lv int
	var LogQ []int
	LogDefaultScale = 48

	// SAFE-Guard for overflow
	q0 := []int{52, 48}
	qiSlotsToCoeffs := []int{48, 48, 48}
	qiCircuits := []int{55, 48, 55}
	for i := 0; i < radix_params.lazy_iter; i++ {
		qiCircuits = append([]int{48}, qiCircuits...)
	}

	var qiLookUpTable []int
	qiLookUpTable = []int{48, 48, 48, 48, 48, 48}

	qiEvalMod := []int{52, 52, 52, 52, 52, 52, 52, 52}
	qiCoeffsToSlots := []int{52, 52, 52}

	LogQ = append(q0, qiSlotsToCoeffs...)
	LogQ = append(LogQ, qiCircuits...)

	// start level
	Lv = len(LogQ)
	LogQ = append(LogQ, qiLookUpTable...)
	LogQ = append(LogQ, qiEvalMod...)
	LogQ = append(LogQ, qiCoeffsToSlots...)

	// ===== Debug print =====
	sum := func(s []int) (t int) {
		for _, v := range s {
			t += v
		}
		return
	}

	fmt.Println("========== Modulus Chain ==========")
	fmt.Printf("lazy_iter        : %d\n", radix_params.lazy_iter)
	fmt.Printf("q0               : %v (len=%d, bits=%d)\n", q0, len(q0), sum(q0))
	fmt.Printf("SlotsToCoeffs    : %v (len=%d, bits=%d)\n", qiSlotsToCoeffs, len(qiSlotsToCoeffs), sum(qiSlotsToCoeffs))
	fmt.Printf("Circuits         : %v (len=%d, bits=%d)\n", qiCircuits, len(qiCircuits), sum(qiCircuits))
	fmt.Printf("LookUpTable      : %v (len=%d, bits=%d)\n", qiLookUpTable, len(qiLookUpTable), sum(qiLookUpTable))
	fmt.Printf("EvalMod          : %v (len=%d, bits=%d)\n", qiEvalMod, len(qiEvalMod), sum(qiEvalMod))
	fmt.Printf("CoeffsToSlots    : %v (len=%d, bits=%d)\n", qiCoeffsToSlots, len(qiCoeffsToSlots), sum(qiCoeffsToSlots))
	fmt.Println("-----------------------------------")
	fmt.Printf("LogQ             : %v\n", LogQ)
	fmt.Printf("#moduli          : %d\n", len(LogQ))
	fmt.Printf("Start level (Lv) : %d\n", Lv)
	fmt.Printf("Total logQ       : %d bits\n", sum(LogQ))
	fmt.Println("===================================")

	var err error
	var params ckks.Parameters

	params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            LogN,                      // Log2 of the ring degree
		LogQ:            LogQ,                      // Log2 of the ciphertext modulus
		LogP:            []int{52, 52, 52, 52, 52}, // Log2 of the key-switch auxiliary prime moduli
		LogDefaultScale: LogDefaultScale,           // Log2 of the scale
		Xs:              ring.Ternary{H: 32768},
	})

	if err != nil {
		panic(err)
	}

	//====================================
	//=== BOOTSTRAPPING PARAMETERS ===
	//====================================

	// CoeffsToSlots parameters (homomorphic encoding)
	CtS_scale := *big.NewFloat(float64(0.5)) // We use real bootstrapping
	CoeffsToSlotsParameters := dft.MatrixLiteral{
		Type:         dft.HomomorphicEncode,
		Format:       dft.RepackImagAsReal, // Returns the real and imaginary part into separate ciphertexts
		LogSlots:     params.LogMaxSlots(),
		LevelQ:       params.MaxLevelQ(),
		LevelP:       params.MaxLevelP(),
		LogBSGSRatio: 1,
		Scaling:      &CtS_scale,
		Levels:       []int{1, 1, 1}, //qiCoeffsToSlots
	}

	// Parameters of the homomorphic modular reduction x mod 1
	var Mod1ParametersLiteral mod1.ParametersLiteral
	Mod1ParametersLiteral = mod1.ParametersLiteral{
		LevelQ:          params.MaxLevel() - CoeffsToSlotsParameters.Depth(true),
		LogScale:        52,               // Matches qiEvalMod
		Mod1Type:        mod1.CosDiscrete, // Multi-interval Chebyshev interpolation
		Mod1Degree:      31,               // Depth iter_num
		DoubleAngle:     3,                // Depth 3
		K:               16,               // With EphemeralSecretWeight = 32 and 2^{155 slots, ensures < 2^{-138.7} failure probability
		LogMessageRatio: 4,                // q/|m| = 2^10
		Mod1InvDegree:   0,                // Depth 0
	}

	// SlotsToCoeffs parameters (homomorphic decoding)
	SlotsToCoeffsParameters := dft.MatrixLiteral{
		Type:         dft.HomomorphicDecode,
		LogSlots:     params.LogMaxSlots(),
		LogBSGSRatio: 1,
		LevelP:       params.MaxLevelP(),
		Levels:       []int{1, 1, 1}, // qiSlotsToCoeffs
	}

	SlotsToCoeffsParameters.LevelQ = 4

	// Custom bootstrapping.Parameters.
	// All fields are public and can be manually instantiated.
	btpParams := bootstrapping.Parameters{
		ResidualParameters:      params,
		BootstrappingParameters: params,
		SlotsToCoeffsParameters: SlotsToCoeffsParameters,
		Mod1ParametersLiteral:   Mod1ParametersLiteral,
		CoeffsToSlotsParameters: CoeffsToSlotsParameters,
		EphemeralSecretWeight:   32, // > 128bit secure for LogN=16 and LogQP = 1155
		CircuitOrder:            bootstrapping.DecodeThenModUp,
	}

	// We pring some information about the bootstrapping parameters (which are identical to the residual parameters in this example).
	// We can notably check that the LogQP of the bootstrapping parameters is smaller than 1555, which ensures
	// 128-bit of security as explained above.
	fmt.Printf("Bootstrapping parameters: logN=%d, logSlots=%d, H(%d; %d), sigma=%f, logQP=%f, levels=%d, scale=2^%d\n",
		btpParams.BootstrappingParameters.LogN(),
		btpParams.BootstrappingParameters.LogMaxSlots(),
		btpParams.BootstrappingParameters.XsHammingWeight(),
		btpParams.EphemeralSecretWeight,
		btpParams.BootstrappingParameters.Xe(),
		btpParams.BootstrappingParameters.LogQP(),
		btpParams.BootstrappingParameters.QCount(),
		btpParams.BootstrappingParameters.LogDefaultScale())

	// Scheme context and keys
	kgen := rlwe.NewKeyGenerator(params)

	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	encoder := ckks.NewEncoder(params)
	decryptor := rlwe.NewDecryptor(params, sk)
	encryptor := rlwe.NewEncryptor(params, pk)

	//fmt.Print("Generating bootstrapping evaluation keys...")
	evk, _, err := btpParams.GenEvaluationKeys(sk)
	if err != nil {
		panic(err)
	}

	var eval *ckks.Evaluator
	eval = ckks.NewEvaluator(params, evk)
	rot := -1 * params.MaxSlots() / slice_length

	//fmt.Println(Compute_Shift_Index(params, slice_length, params.MaxSlots()/slice_length))
	galEls := []uint64{
		// The galois element for the cyclic rotations by 55positions to the left.
		params.GaloisElement(rot),
		params.GaloisElement(params.MaxSlots() / slice_length * slice_length / 4),
		params.GaloisElement(params.MaxSlots() / slice_length * (slice_length/4 - 1)),
		params.GaloisElement(params.MaxSlots() / 2),
		// The galois element for the complex conjugatation.
		params.GaloisElementForComplexConjugation(),
	}

	//params.GaloisElementOrderTwoOrthogonalSubgroup()

	shift_idx := Compute_Shift_Index(params, slice_length, params.MaxSlots()/slice_length)

	for i := 0; i < len(shift_idx); i++ {
		galEls = append(galEls, []uint64{params.GaloisElement(shift_idx[i])}...)
	}

	//fmt.Println("Required Galois Key")
	//fmt.Println(galEls)

	// We then generate the `rlwe.GaloisKey`s element that corresponds to these galois elements.
	// And we update the evaluator's `rlwe.EvaluationKeySet` with the new keys.
	eval = eval.WithKey(rlwe.NewMemEvaluationKeySet(rlk, kgen.GenGaloisKeysNew(galEls, sk)...))

	// Instantiates the bootstrapper
	var btp *bootstrapping.Evaluator

	if btp, err = bootstrapping.NewEvaluator(btpParams, evk); err != nil {
		panic(err)
	}

	var cc Context
	cc.params = &params
	cc.encoder = encoder
	cc.encryptor = encryptor
	cc.decryptor = decryptor
	cc.eval = eval
	cc.btp = btp

	//==============================================
	//=== 1) Encrypt Large Integer  ================
	//==============================================

	batch := int(float64(params.MaxSlots()) / float64(slice_length))
	fmt.Printf("Large Integer parameter : bit_length=%d, batch=%d, lazycarry_iter=%d, carry_iter=%d", bit_length, batch, radix_params.lazy_iter, radix_params.logK)
	fmt.Println()
	fmt.Println()

	fmt.Printf("make DFT matrix...")
	if file, _ := FileExists("precom/DFT_" + strconv.Itoa(slice_length) + "B16"); file == false {
		// Generate FFT matrix
		Normalized_DFT := GenerateSpecialNormalizedDFT(slice_length)
		//Normalized_DFT = MatrixPadding(Normalized_DFT, slice_length, params.MaxSlots())
		//Normalized_DFT2 = MatrixPadding(Normalized_DFT2, slice_length, params.MaxSlots())
		Twisted_DFT := TwistedMatrix(Normalized_DFT, slice_length, params.MaxSlots())
		//Twisted_DFT2 := TwistedMatrix(Normalized_DFT2, slice_length, params.MaxSlots())
		Diag_DFT := Row_To_Diagonal(Twisted_DFT)
		//Diag_DFT2 := Row_To_Diagonal(Twisted_DFT2)
		//fmt.Println(Diag_DFT2[0][0])
		Normalized_DFT = nil
		Twisted_DFT = nil
		runtime.GC()

		Plain_DFT := BSGS_plain_Gen(params, slice_length, batch, Lv-1, Diag_DFT, cc)

		if err := SavePlaintextMap("precom/DFT_"+strconv.Itoa(slice_length)+"B16", Plain_DFT); err != nil {
			panic(err)
		}

		if err != nil {
			panic(err)
		}
		Diag_DFT = nil
		runtime.GC()
	}

	Plain_DFT, err := LoadPlaintextMap("precom/DFT_"+strconv.Itoa(slice_length)+"B16", params)

	if file, _ := FileExists("precom/InvDFT_" + strconv.Itoa(slice_length) + "B16"); file == false {
		InvDFT := GenerateNormalizedInvDFT_with_masking(slice_length)
		//InvDFT = MatrixPadding(InvDFT, slice_length, params.MaxSlots())
		Twisted_InvDFT := TwistedMatrix(InvDFT, slice_length, params.MaxSlots())
		Diag_InvDFT := Row_To_Diagonal(Twisted_InvDFT)

		InvDFT = nil
		Twisted_InvDFT = nil
		runtime.GC()

		Plain_InvDFT := BSGS_plain_Gen(params, slice_length, batch, Lv-3, Diag_InvDFT, cc)

		if err := SavePlaintextMap("precom/InvDFT_"+strconv.Itoa(slice_length)+"B16", Plain_InvDFT); err != nil {
			panic(err)
		}

		if err != nil {
			panic(err)
		}
		Diag_InvDFT = nil
		runtime.GC()
	}
	fmt.Println("Done!")

	Plain_InvDFT, err := LoadPlaintextMap("precom/InvDFT_"+strconv.Itoa(slice_length)+"B16", params)

	if setting == true {
		return
	}

	//==============================================
	//=== 2) Mult Large Integer : MultPoly =========
	//==============================================

	iter_num := 1
	lazy_time := make([]float64, iter_num)
	exact_time := make([]float64, iter_num)
	min_err := make([]float64, iter_num)
	avg_err := make([]float64, iter_num)

	//warmupCPU(30 * time.Second)
	fmt.Println("=== Multiplication Start ===")
	for i := 0; i < iter_num; i++ {

		values1_high_real := make([]*big.Float, params.MaxSlots()/slice_length)
		for i := 0; i < len(values1_high_real); i++ {
			values1_high_real[i] = SampleFixedPrecision(2048)
			// if i == 0 {
			// 	values1_high_real[i] = big.NewFloat(0.5)
			// }
			// if i == 1 {
			// 	values1_high_real[i] = big.NewFloat(-0.5)
			// 	//fmt.Println(values1_high_real[i])
			// }
			// if i == 2 {
			// 	values1_high_real[i] = big.NewFloat(0.5)
			// 	//fmt.Println(values1_high_real[i])
			// }
			// if i == 3 {
			// 	values1_high_real[i] = big.NewFloat(-0.5)
			// 	//fmt.Println(values1_high_real[i])
			// }
			//fmt.Println(i, values1_high_real[i])
		}

		values2_high_real := make([]*big.Float, params.MaxSlots()/slice_length)
		for i := 0; i < len(values2_high_real); i++ {
			values2_high_real[i] = SampleFixedPrecision(2048)
			// if i == 0 {
			// 	values2_high_real[i] = big.NewFloat(0.5)
			// }
			// if i == 1 {
			// 	values2_high_real[i] = big.NewFloat(0.5)
			// 	//fmt.Println(values2_high_real[i])
			// }
			// if i == 2 {
			// 	values2_high_real[i] = big.NewFloat(-0.5)
			// 	//fmt.Println(values1_high_real[i])
			// }
			// if i == 3 {
			// 	values2_high_real[i] = big.NewFloat(-0.5)
			// 	//fmt.Println(values1_high_real[i])
			// }
			//fmt.Println(i, values2_high_real[i])
		}

		values_rtn_high_real := make([]*big.Float, params.MaxSlots()/slice_length)
		for i := 0; i < len(values2_high_real); i++ {
			values_rtn_high_real[i] = big.NewFloat(1.0).SetPrec(2048).Add(values1_high_real[i], values2_high_real[i])
			//values_rtn_high_real[i] = big.NewFloat(1.0).SetPrec(2048).Mul(values1_high_real[i], values2_high_real[i])
			if i == 0 {
				//fmt.Println(values_rtn_high_real[i])
			}
		}

		// == 1-1 : packing
		scale := big.NewInt(2.0).Exp(big.NewInt(int64(base)), big.NewInt(int64(slice_length)/4-1), nil)
		mod := big.NewInt(2.0).Exp(big.NewInt(int64(base)), big.NewInt(int64(slice_length)/2), nil)
		sign := big.NewInt(2.0).Exp(big.NewInt(int64(base)), big.NewInt(int64(slice_length)/2-1), nil)
		//fmt.Println(scale)

		value1 := make([]complex128, params.MaxSlots())
		for i := 0; i < params.MaxSlots()/slice_length; i++ {
			scaled_real := MulAndRound(values1_high_real[i], scale)
			scaled_real = big.NewInt(1).Mod(scaled_real, mod)
			digits := DecomposeBaseFixed(scaled_real, big.NewInt(int64(base)), slice_length)
			if i == 0 {
				//fmt.Println(scaled_real)
			}
			for j := 0; j < slice_length; j++ {
				//fmt.Println(j, digits[j])
				value1[i*slice_length+j] = complex(float64(digits[j]), 0)
				if i == 0 {
					//value1[i*slice_length+j] = complex(1, 0)
				}
				if i == 0 {
					//fmt.Println(j, digits[j])
				}
			}
		}

		value2 := make([]complex128, params.MaxSlots())
		for i := 0; i < params.MaxSlots()/slice_length; i++ {
			scaled_real := MulAndRound(values2_high_real[i], scale)
			scaled_real = big.NewInt(1).Mod(scaled_real, mod)
			//fmt.Println(scaled_real)
			if i == 0 {
				//fmt.Println(scaled_real)
			}
			digits := DecomposeBaseFixed(scaled_real, big.NewInt(int64(base)), slice_length)
			for j := 0; j < slice_length; j++ {
				if i == 0 {
					//fmt.Println(j, digits[j])
				}
				value2[i*slice_length+j] = complex(float64(digits[j]), 0)
			}
		}

		// Encrypt value 1
		twisted_values1 := TwistedVec(value1, slice_length, params.MaxSlots())
		twisted_values2 := TwistedVec(value2, slice_length, params.MaxSlots())

		plaintext1 := ckks.NewPlaintext(params, Lv-1)
		if err := encoder.Encode(twisted_values1, plaintext1); err != nil {
			panic(err)
		}

		ciphertext1, err := encryptor.EncryptNew(plaintext1)
		if err != nil {
			panic(err)
		}

		// Encrypt value 2
		plaintext2 := ckks.NewPlaintext(params, Lv-1)
		if err := encoder.Encode(twisted_values2, plaintext2); err != nil {
			panic(err)
		}

		ciphertext2, err := encryptor.EncryptNew(plaintext2)
		if err != nil {
			panic(err)
		}

		valuesWant := make([]complex128, params.MaxSlots())
		valuesTest := make([]complex128, params.MaxSlots())
		valuesWant_big := make([]*big.Int, batch)
		valuesTest_big := make([]*big.Int, batch)

		var ciphertext *rlwe.Ciphertext
		opt := 0
		if opt == 0 {
			add_time := time.Now()

			// lazy mult
			ciphertext, _ = cc.eval.AddNew(ciphertext1, ciphertext2)

			// excat mult
			ciphertext = LazyCarry2Carry(ciphertext, radix_params, cc)

			// ciphertext, _ = cc.eval.AddNew(ciphertext, ciphertext)

			// // excat mult
			// ciphertext = LazyCarry2Carry(ciphertext, radix_params, cc)
			// //PrintDebug(slice_length, params, ciphertext_add, valuesWant, decryptor, encoder)
			// //ciphertext_add = ciphertext1

			// cc.eval.Rotate(ciphertext, params.MaxSlots()/slice_length*slice_length/4, ciphertext)
			// mask := make([]complex128, params.MaxSlots())
			// for i := 0; i < params.MaxSlots()/slice_length; i++ {
			// 	for j := 0; j < slice_length; j++ {
			// 		if j < slice_length/4 {
			// 			mask[i*slice_length+j] = complex(1, 0)
			// 		}
			// 	}
			// }
			// twisted_mask := TwistedVec(mask, slice_length, params.MaxSlots())
			// cc.eval.MulRelin(ciphertext, twisted_mask, ciphertext)
			// cc.eval.Rescale(ciphertext, ciphertext)

			total_elapsed := time.Since(add_time)
			fmt.Println("addition time : ", total_elapsed)
			fmt.Println()

		} else if opt == 1 {
			mult_time := time.Now()
			//var debug *rlwe.Ciphertext
			// lazy mult
			ciphertext = LazyMult(ciphertext1, ciphertext2, Plain_DFT, Plain_InvDFT, radix_params, cc)
			ciphertext = LazyCarry2Carry(ciphertext, radix_params, cc)

			scale_half := make([]complex128, params.MaxSlots())
			for i := 0; i < params.MaxSlots()/slice_length; i++ {
				for j := 0; j < slice_length; j++ {
					if j == slice_length/4-2 {
						scale_half[i*slice_length+j] = complex(8, 0)
					}
				}
			}

			twisted_s := TwistedVec(scale_half, slice_length, params.MaxSlots())
			ciphertext, _ = cc.eval.AddNew(ciphertext, twisted_s)

			ciphertext = LazyCarry2Carry(ciphertext, radix_params, cc)
			ciphertext = Cleaning_with_vectorized_evaluation(ciphertext, radix_params, cc)
			cc.eval.Rotate(ciphertext, params.MaxSlots()/slice_length*(slice_length/4-1), ciphertext)
			mask := make([]complex128, params.MaxSlots())
			for i := 0; i < params.MaxSlots()/slice_length; i++ {
				for j := 0; j < slice_length; j++ {
					if j < slice_length/4 {
						mask[i*slice_length+j] = complex(1, 0)
					}
				}
			}

			twisted_mask := TwistedVec(mask, slice_length, params.MaxSlots())
			cc.eval.MulRelin(ciphertext, twisted_mask, ciphertext)
			cc.eval.Rescale(ciphertext, ciphertext)

			//debug = ciphertext.CopyNew()

			sign_vec := make([]complex128, params.MaxSlots())
			for i := 0; i < params.MaxSlots()/slice_length; i++ {
				for j := 0; j < slice_length; j++ {
					if j == slice_length/4-1 {
						sign_vec[i*slice_length+j] = complex(8, 0)
					}
				}
			}
			twisted_sign_vec := TwistedVec(sign_vec, slice_length, params.MaxSlots())

			mod_vec := make([]complex128, params.MaxSlots())
			for i := 0; i < params.MaxSlots()/slice_length; i++ {
				for j := 0; j < slice_length; j++ {
					if j == slice_length/4 {
						mod_vec[i*slice_length+j] = complex(1, 0)
					}
				}
			}

			twisted_mod_vec := TwistedVec(mod_vec, slice_length, params.MaxSlots())

			// mod2_vec := make([]complex128, params.MaxSlots())
			// for i := 0; i < params.MaxSlots()/slice_length; i++ {
			// 	for j := 0; j < slice_length; j++ {
			// 		if j == slice_length/2-1 {
			// 			mod2_vec[i*slice_length+j] = complex(1, 0)
			// 		}
			// 	}
			// }
			// twisted_mod2_vec := TwistedVec(mod2_vec, slice_length, params.MaxSlots())

			//temp := ciphertext.CopyNew()
			one := ComparisonPlain(ciphertext, twisted_sign_vec, radix_params, cc)
			//var debug *rlwe.Ciphertext
			//debug = one.CopyNew()
			//one = Cleaning_with_vectorized_evaluation(one, radix_params, cc)
			MAX_SLOT := cc.params.MaxSlots()
			BATCH := MAX_SLOT / slice_length
			for i := 0; i < radix_params.logK; i++ {
				var temp *rlwe.Ciphertext
				temp, err = cc.eval.RotateNew(one, -1*(1<<i)*BATCH)
				if err != nil {
					fmt.Println(err)
				}

				cc.eval.Add(one, temp, one)
			}

			//cc.eval.Sub(one, 1, one)
			//cc.eval.Mul(one, -1, one)

			x, _ := cc.eval.MulRelinNew(one, twisted_mod_vec)
			cc.eval.Rescale(x, x)

			ciphertext = Sub(ciphertext, x, radix_params, cc, "")
			//ciphertext = debug
			//ciphertext = one
			//cc.eval.MulRelin(one, twisted_mod2_vec, one)
			//cc.eval.Rescale(one, one)

			//cc.eval.Add(ciphertext, one, ciphertext)

			//PrintDebug(slice_length, params, ciphertext_add, valuesWant, decryptor, encoder)
			//ciphertext_add = ciphertext1
			//ciphertext = debug

			total_elapsed := time.Since(mult_time)
			fmt.Println("multiplication time : ", total_elapsed)
			fmt.Println()
		} else {
			t := time.Now()
			ciphertext = BitShift(base, slice_length, ciphertext1, cc, 2)
			fmt.Println(time.Since(t))
		}

		// ciphertext_mult, _ := cc.eval.AddNew(ciphertext1, ciphertext2)
		// //PrintDebug(slice_length, params, ciphertext, valuesWant, decryptor, encoder)

		// // excat mult
		// ciphertext_mult = LazyCarry2Carry(ciphertext_mult, radix_params, cc)

		// total_elapsed := time.Since(add_time)

		//==============================================
		//=== 53 Decrypt and Validation ================
		//==============================================

		plaintext := decryptor.DecryptNew(ciphertext)
		encoder.Decode(plaintext, valuesTest)

		valuesTest = InvTwistedVec(valuesTest, slice_length, params.MaxSlots())

		// Time
		fmt.Println("=== Experiment Result ===")

		// Accuracy

		values_Test_high_real := make([]*big.Float, params.MaxSlots()/slice_length)
		// Want : Convert Radix int to Int
		for i := 0; i < params.MaxSlots()/slice_length; i++ {
			int_arr := make([]int, slice_length)
			val_arr := make([]complex128, slice_length)
			for j := 0; j < slice_length; j++ {
				int_arr[j] = int(math.Round(real(valuesTest[i*slice_length+j])))
				val_arr[j] = valuesTest[i*slice_length+j]
			}

			//fmt.Println(int_arr)
			//fmt.Println(val_arr)

			scale_real := ComposeBase(int_arr, big.NewInt(int64(base)))
			//fmt.Println(scale_real)
			// scale_real >= mod 이면 mod를 뺌
			scale_real.Mod(scale_real, mod)
			//fmt.Println(scale_real)
			if scale_real.Cmp(sign) >= 0 {
				scale_real.Sub(scale_real, mod)
				//scale_real.Sub(scale_real, scale)
			}
			//fmt.Println(scale_real)

			values_Test_high_real[i] = IntDivScale(scale_real, scale)
			//fmt.Println(values_rtn_high_real[i], values_Test_high_real[i])
			if i == 0 {
				// fmt.Println(values_Test_high_real[i])
			}
		}

		for i := 0; i < params.MaxSlots()/slice_length; i++ {

			//fmt.Println(Log2Error(values_rtn_high_real[i], values_Test_high_real[i], 128))
		}

		sumAbsErr := new(big.Float).SetPrec(2048).SetFloat64(0.0)

		minErr := new(big.Float).SetPrec(2048)
		maxErr := new(big.Float).SetPrec(2048)

		count := params.MaxSlots() / slice_length

		for i := 0; i < count; i++ {

			// Absolute error: e_i = |x_hat_i - x_i|
			absErr := new(big.Float).SetPrec(2048).Sub(
				values_rtn_high_real[i],
				values_Test_high_real[i],
			)

			if absErr.Sign() < 0 {
				absErr.Neg(absErr)
			}

			sumAbsErr.Add(sumAbsErr, absErr)

			if i == 0 {
				minErr.Set(absErr)
				maxErr.Set(absErr)
			} else {
				if absErr.Cmp(minErr) < 0 {
					minErr.Set(absErr)
				}

				if absErr.Cmp(maxErr) > 0 {
					maxErr.Set(absErr)
				}
			}

			//fmt.Printf(
			//	"i=%d, absErr=%s\n",
			//	i,
			//	absErr.Text('e', 50),
			//)
		}

		countFloat := new(big.Float).
			SetPrec(2048).
			SetInt64(int64(count))

		// Mean absolute error
		meanErr := new(big.Float).
			SetPrec(2048).
			Quo(sumAbsErr, countFloat)

		// log2(meanErr)
		var log2MeanErr float64

		if meanErr.Sign() == 0 {
			log2MeanErr = math.Inf(-1)
		} else {
			mant := new(big.Float).SetPrec(2048)
			exp := meanErr.MantExp(mant)

			mantFloat64, _ := mant.Float64()

			log2MeanErr =
				math.Log2(math.Abs(mantFloat64)) +
					float64(exp)
		}

		// log2(maxErr)
		var log2MaxErr float64

		if maxErr.Sign() == 0 {
			log2MaxErr = math.Inf(-1)
		} else {
			mant := new(big.Float).SetPrec(2048)
			exp := maxErr.MantExp(mant)

			mantFloat64, _ := mant.Float64()

			log2MaxErr =
				math.Log2(math.Abs(mantFloat64)) +
					float64(exp)
		}

		// log2(minErr)
		var log2MinErr float64

		if minErr.Sign() == 0 {
			log2MinErr = math.Inf(-1)
		} else {
			mant := new(big.Float).SetPrec(2048)
			exp := minErr.MantExp(mant)

			mantFloat64, _ := mant.Float64()

			log2MinErr =
				math.Log2(math.Abs(mantFloat64)) +
					float64(exp)
		}

		fmt.Printf("\n")
		fmt.Printf("Mean Error : %s\n", meanErr.Text('e', 50))
		fmt.Printf("Max Error  : %s\n", maxErr.Text('e', 50))
		fmt.Printf("Min Error  : %s\n", minErr.Text('e', 50))

		fmt.Printf("\n")
		fmt.Printf("log2(Mean Error) : %.10f\n", log2MeanErr)
		fmt.Printf("log2(Max Error)  : %.10f\n", log2MaxErr)
		fmt.Printf("log2(Min Error)  : %.10f\n", log2MinErr)

		fmt.Printf("\n")
		fmt.Printf("-log2(Mean Error) : %.10f bits\n", -log2MeanErr)
		fmt.Printf("-log2(Max Error)  : %.10f bits\n", -log2MaxErr)
		fmt.Printf("-log2(Min Error)  : %.10f bits\n", -log2MinErr)
		acc := 0.0
		for i := 0; i < len(valuesTest_big); i++ {
			if valuesTest_big[i].Cmp(valuesWant_big[i]) == 0 {
				acc += 1.0
			}
		}
		fmt.Printf("Large Integer accuracy : %0.4f\n", acc/float64(batch))

		//fmt.Print("Test(Big) : ")
		//fmt.Printf("[%d, %d, %d, ... , %d]\n", valuesTest_big[0], valuesTest_big[1], valuesTest_big[2], valuesTest_big[len(valuesTest_big)-1])
		//fmt.Print("Want(Big) : ")
		//fmt.Printf("[%d, %d, %d, ... , %d]\n", valuesWant_big[0], valuesWant_big[1], valuesWant_big[2], valuesWant_big[len(valuesWant_big)-1])

		// precision

		//lazy_time[i] = float64(lazy_elapsed.Seconds())
		//exact_time[i] = float64(total_elapsed.Seconds())
		_, min, avg := ComputePrec(base, slice_length, params, ciphertext, valuesWant, decryptor, encoder)

		min_err[i] = min
		avg_err[i] = avg

		fmt.Println()
	}

	var lazy_mean, exact_mean, min_mean, avg_mean float64
	for i := 0; i < iter_num; i++ {
		lazy_mean += lazy_time[i]
		exact_mean += exact_time[i]
		min_mean += min_err[i]
		avg_mean += avg_err[i]
	}
	lazy_mean /= float64(iter_num)
	exact_mean /= float64(iter_num)
	min_mean /= float64(iter_num)
	avg_mean /= float64(iter_num)

	fmt.Println("lazy lat. : ", lazy_mean, " s")
	fmt.Println("lazy amot.. : ", 1000*lazy_mean/float64(batch), "ms")
	fmt.Println("exact lat. : ", exact_mean, " s")
	fmt.Println("exact amot. : ", 1000*exact_mean/float64(batch), " ms")
	fmt.Println("min err : ", min_mean)
	fmt.Println("avg err : ", avg_mean)

}
