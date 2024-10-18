package e2etest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"sync"
	"testing"
	"time"

	staking "github.com/babylonlabs-io/babylon/btcstaking"
	"github.com/babylonlabs-io/babylon/btctxformatter"
	btcctypes "github.com/babylonlabs-io/babylon/x/btccheckpoint/types"
	btckpttypes "github.com/babylonlabs-io/babylon/x/btccheckpoint/types"
	checkpointingtypes "github.com/babylonlabs-io/babylon/x/checkpointing/types"
	"github.com/babylonlabs-io/vigilante/e2etest/container"

	btcstypes "github.com/babylonlabs-io/babylon/x/btcstaking/types"

	"cosmossdk.io/errors"
	sdkmath "cosmossdk.io/math"
	"github.com/babylonlabs-io/babylon/app"
	bbn "github.com/babylonlabs-io/babylon/app"
	"github.com/babylonlabs-io/babylon/client/config"
	bncfg "github.com/babylonlabs-io/babylon/client/config"
	"github.com/babylonlabs-io/babylon/client/query"
	"github.com/babylonlabs-io/babylon/crypto/bip322"
	asig "github.com/babylonlabs-io/babylon/crypto/schnorr-adaptor-signature"
	"github.com/babylonlabs-io/babylon/testutil/datagen"
	bbntypes "github.com/babylonlabs-io/babylon/types"
	btclctypes "github.com/babylonlabs-io/babylon/x/btclightclient/types"
	bstypes "github.com/babylonlabs-io/babylon/x/btcstaking/types"
	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/btcec/v2/schnorr"
	"github.com/btcsuite/btcd/btcjson"
	"github.com/btcsuite/btcd/btcutil"
	"github.com/btcsuite/btcd/btcutil/psbt"
	"github.com/btcsuite/btcd/chaincfg/chainhash"
	"github.com/btcsuite/btcd/txscript"
	"github.com/btcsuite/btcd/wire"
	"github.com/cometbft/cometbft/crypto/tmhash"
	rpchttp "github.com/cometbft/cometbft/rpc/client/http"
	"github.com/cosmos/cosmos-sdk/client"
	"github.com/cosmos/cosmos-sdk/client/tx"
	cryptotypes "github.com/cosmos/cosmos-sdk/crypto/types"
	"github.com/cosmos/cosmos-sdk/testutil/testdata"
	"github.com/cosmos/cosmos-sdk/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/tx/signing"
	xauthsigning "github.com/cosmos/cosmos-sdk/x/auth/signing"
	banktypes "github.com/cosmos/cosmos-sdk/x/bank/types"
	stakingtypes "github.com/cosmos/cosmos-sdk/x/staking/types"
	"github.com/cosmos/relayer/v2/relayer/chains/cosmos"
	pv "github.com/cosmos/relayer/v2/relayer/provider"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

const (
	chainId = "chain-test"
)

func BuildTxWithMsgs() {
	tmpApp := app.NewTmpBabylonApp()

	// Create a new TxBuilder.
	txBuilder := tmpApp.TxConfig().NewTxBuilder()

	txBuilder.SetGasLimit(1000000)
}

type BabylonSender struct {
	PrvKey         cryptotypes.PrivKey
	PubKey         cryptotypes.PubKey
	BabylonAddress sdk.AccAddress
	t              *testing.T
	ClientContext  client.Context
}

func NewBabylonSender(
	t *testing.T,
	clientCtx client.Context) *BabylonSender {
	prvKey, pubKey, addr := testdata.KeyTestPubAddr()

	return &BabylonSender{
		PrvKey:         prvKey,
		PubKey:         pubKey,
		BabylonAddress: addr,
		t:              t,
		ClientContext:  clientCtx,
	}
}

func (bs *BabylonSender) SendMsgs(msgs []sdk.Msg) {
	account, err := bs.ClientContext.AccountRetriever.GetAccount(bs.ClientContext, bs.BabylonAddress)
	require.NoError(bs.t, err)

	// Create a new TxBuilder.
	txBuilder := bs.ClientContext.TxConfig.NewTxBuilder()
	txBuilder.SetFeePayer(bs.BabylonAddress)
	txBuilder.SetGasLimit(1000000)
	txBuilder.SetMsgs(msgs...)

	seqNr := account.GetSequence() + 1
	sigV2 := signing.SignatureV2{
		PubKey: bs.PubKey,
		Data: &signing.SingleSignatureData{
			SignMode:  signing.SignMode_SIGN_MODE_DIRECT,
			Signature: nil,
		},
		Sequence: seqNr,
	}

	txBuilder.SetSignatures(sigV2)

	signerData := xauthsigning.SignerData{
		ChainID:       bs.ClientContext.ChainID,
		AccountNumber: account.GetAccountNumber(),
		Sequence:      seqNr,
	}

	sigV2full, err := tx.SignWithPrivKey(
		context.Background(),
		signing.SignMode_SIGN_MODE_DIRECT, signerData,
		txBuilder, bs.PrvKey, bs.ClientContext.TxConfig, seqNr)
	require.NoError(bs.t, err)
	txBuilder.SetSignatures(sigV2full)

	txBytes, err := bs.ClientContext.TxConfig.TxEncoder()(txBuilder.GetTx())
	require.NoError(bs.t, err)

	resp, err := bs.ClientContext.BroadcastTxSync(txBytes)
	require.NoError(bs.t, err)
	require.NotNil(bs.t, resp)

	fmt.Println(resp.Code)
}

// func generateAndInsertHeader(t *testing.T, tm *TestManager, height uint32) {
// 	// Generate a new block header and insert it into the blockchain.
// 	header := tm.BitcoindHandler.GenerateBlocks(1)
// 	headerHash :=

// 		tm.BTCClient.GetBlockHeader()

// 	err := tm.BabylonClient.InsertHeader(header)
// 	require.NoError(t, err)
// }

func (tm *TestManager) fundAllParties(
	t *testing.T,
	senders []*SenderWithBabylonClient,
) {

	fundingAccount := tm.BabylonClient.MustGetAddr()
	fundingAddress := sdk.MustAccAddressFromBech32(fundingAccount)

	var msgs []sdk.Msg

	for _, sender := range senders {
		msg := banktypes.NewMsgSend(fundingAddress, sender.BabylonAddress, types.NewCoins(types.NewInt64Coin("ubbn", 100000000)))
		msgs = append(msgs, msg)
	}

	resp, err := tm.BabylonClient.ReliablySendMsgs(
		context.Background(),
		msgs,
		[]*errors.Error{},
		[]*errors.Error{},
	)
	require.NoError(t, err)
	require.NotNil(t, resp)
}

func senders(stakers []*BTCStaker) []*SenderWithBabylonClient {
	var senders []*SenderWithBabylonClient

	for _, staker := range stakers {
		stakerCp := staker
		senders = append(senders, stakerCp.client)
	}
	return senders
}

func TestXxx(t *testing.T) {
	// tmpApp := app.NewTmpBabylonApp()
	// more outputs to handle more stakers
	numMatureOutputs := uint32(500)
	tm := StartManager(t, numMatureOutputs, 10)
	defer tm.Stop(t)

	cpSender := babylonClientSender(t, "node0", tm.Config.Babylon.RPCAddr, tm.Config.Babylon.GRPCAddr)
	headerSender := babylonClientSender(t, "headerreporter", tm.Config.Babylon.RPCAddr, tm.Config.Babylon.GRPCAddr)
	vigilanteSender := babylonClientSender(t, "vigilante", tm.Config.Babylon.RPCAddr, tm.Config.Babylon.GRPCAddr)

	tm.fundAllParties(t, []*SenderWithBabylonClient{cpSender, headerSender, vigilanteSender})

	fpResp, fpInfo, err := cpSender.CreateFinalityProvider(t)
	fmt.Println(fpResp)
	fmt.Println(err)

	numStakers := 10

	var stakers []*BTCStaker
	for i := 0; i < numStakers; i++ {
		stakerSender := babylonClientSender(t, fmt.Sprintf("staker%d", i), tm.Config.Babylon.RPCAddr, tm.Config.Babylon.GRPCAddr)
		staker := NewBTCStaker(t, tm, stakerSender, fpInfo.BtcPk.MustToBTCPK())
		stakers = append(stakers, staker)
	}

	// fund all stakers
	tm.fundAllParties(t, senders(stakers))

	gen := NewBTCHeaderGenerator(t, tm, headerSender)
	gen.Start()
	defer gen.Stop()

	vig := NewSubReporter(t, tm, vigilanteSender)
	vig.Start()
	defer vig.Stop()

	// start stakers and defer stops
	// TODO: Ideally stakers would start on different times to reduce contention
	// on funding BTC wallet
	for _, staker := range stakers {
		staker.Start()
		defer staker.Stop()
	}

	covenantSender := babylonClientSender(t, "covenant", tm.Config.Babylon.RPCAddr, tm.Config.Babylon.GRPCAddr)
	covenant := NewCovenantEmulator(t, tm, container.CovnentPrivKey, covenantSender)
	tm.fundAllParties(t, []*SenderWithBabylonClient{covenantSender})

	covenant.Start()
	defer covenant.Stop()

	time.Sleep(120 * time.Second)
}

type SenderWithBabylonClient struct {
	*Client
	PrvKey         cryptotypes.PrivKey
	PubKey         cryptotypes.PubKey
	BabylonAddress sdk.AccAddress
}

func babylonClientSender(
	t *testing.T,
	keyName string,
	rpcaddr string,
	grpcaddr string) *SenderWithBabylonClient {

	cfg := bncfg.DefaultBabylonConfig()
	cfg.Key = keyName
	cfg.ChainID = chainId
	cfg.KeyringBackend = "memory"
	cfg.RPCAddr = rpcaddr
	cfg.GRPCAddr = grpcaddr
	cfg.GasAdjustment = 3.0

	cl, err := New(&cfg, zap.NewNop())
	require.NoError(t, err)

	prvKey, pubKey, address := testdata.KeyTestPubAddr()

	err = cl.provider.Keybase.ImportPrivKeyHex(
		keyName,
		hex.EncodeToString(prvKey.Bytes()),
		"secp256k1",
	)
	require.NoError(t, err)

	return &SenderWithBabylonClient{
		Client:         cl,
		PrvKey:         prvKey,
		PubKey:         pubKey,
		BabylonAddress: address,
	}
}

type Client struct {
	*query.QueryClient

	provider *cosmos.CosmosProvider
	timeout  time.Duration
	logger   *zap.Logger
	cfg      *config.BabylonConfig
}

func New(
	cfg *config.BabylonConfig, logger *zap.Logger) (*Client, error) {
	var (
		zapLogger *zap.Logger
		err       error
	)

	// ensure cfg is valid
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	// use the existing logger or create a new one if not given
	zapLogger = logger
	if zapLogger == nil {
		zapLogger = zap.NewNop()
	}

	provider, err := cfg.ToCosmosProviderConfig().NewProvider(
		zapLogger,
		"", // TODO: set home path
		true,
		"babylon",
	)
	if err != nil {
		return nil, err
	}

	cp := provider.(*cosmos.CosmosProvider)
	cp.PCfg.KeyDirectory = cfg.KeyDirectory

	// Create tmp Babylon app to retrieve and register codecs
	// Need to override this manually as otherwise option from config is ignored
	encCfg := bbn.GetEncodingConfig()
	cp.Cdc = cosmos.Codec{
		InterfaceRegistry: encCfg.InterfaceRegistry,
		Marshaler:         encCfg.Codec,
		TxConfig:          encCfg.TxConfig,
		Amino:             encCfg.Amino,
	}

	// initialise Cosmos provider
	// NOTE: this will create a RPC client. The RPC client will be used for
	// submitting txs and making ad hoc queries. It won't create WebSocket
	// connection with Babylon node
	err = cp.Init(context.Background())
	if err != nil {
		return nil, err
	}

	// create a queryClient so that the Client inherits all query functions
	// TODO: merge this RPC client with the one in `cp` after Cosmos side
	// finishes the migration to new RPC client
	// see https://github.com/strangelove-ventures/cometbft-client
	c, err := rpchttp.NewWithTimeout(cp.PCfg.RPCAddr, "/websocket", uint(cfg.Timeout.Seconds()))
	if err != nil {
		return nil, err
	}
	queryClient, err := query.NewWithClient(c, cfg.Timeout)
	if err != nil {
		return nil, err
	}

	return &Client{
		queryClient,
		cp,
		cfg.Timeout,
		zapLogger,
		cfg,
	}, nil
}

func ToProviderMsgs(msgs []sdk.Msg) []pv.RelayerMessage {
	relayerMsgs := []pv.RelayerMessage{}
	for _, m := range msgs {
		relayerMsgs = append(relayerMsgs, cosmos.NewCosmosMessage(m, func(signer string) {}))
	}
	return relayerMsgs
}

func (s *SenderWithBabylonClient) SendMsgs(ctx context.Context, msgs []sdk.Msg) (*pv.RelayerTxResponse, error) {
	relayerMsgs := ToProviderMsgs(msgs)
	resp, success, err := s.provider.SendMessages(ctx, relayerMsgs, "")

	if err != nil {
		return nil, err
	}

	if !success {
		return resp, fmt.Errorf("Message send but failed to execute")
	}

	return resp, nil
}

func (s *SenderWithBabylonClient) CreateFinalityProvider(t *testing.T) (*pv.RelayerTxResponse, *bstypes.FinalityProvider, error) {
	var err error
	signerAddr := s.BabylonAddress.String()
	addr := sdk.MustAccAddressFromBech32(signerAddr)

	fpSK, _, err := datagen.GenRandomBTCKeyPair(r)
	require.NoError(t, err)
	btcFp, err := datagen.GenRandomFinalityProviderWithBTCBabylonSKs(r, fpSK, addr)
	require.NoError(t, err)

	commission := sdkmath.LegacyZeroDec()
	msgNewVal := &bstypes.MsgCreateFinalityProvider{
		Addr:        signerAddr,
		Description: &stakingtypes.Description{Moniker: datagen.GenRandomHexStr(r, 10)},
		Commission:  &commission,
		BtcPk:       btcFp.BtcPk,
		Pop:         btcFp.Pop,
	}
	resp, err := s.SendMsgs(context.Background(), []sdk.Msg{msgNewVal})

	if err != nil {
		return resp, nil, err
	}

	return resp, btcFp, nil
}

func (s *SenderWithBabylonClient) InsertBTCHeadersToBabylon(headers []*wire.BlockHeader) (*pv.RelayerTxResponse, error) {
	var headersBytes []bbntypes.BTCHeaderBytes

	for _, h := range headers {
		headersBytes = append(headersBytes, bbntypes.NewBTCHeaderBytesFromBlockHeader(h))
	}

	msg := btclctypes.MsgInsertHeaders{
		Headers: headersBytes,
		Signer:  s.BabylonAddress.String(),
	}

	return s.SendMsgs(context.Background(), []sdk.Msg{&msg})
}

type BTCHeaderGenerator struct {
	t      *testing.T
	tm     *TestManager
	client *SenderWithBabylonClient
	wg     *sync.WaitGroup
	quit   chan struct{}
}

func NewBTCHeaderGenerator(
	t *testing.T,
	tm *TestManager,
	client *SenderWithBabylonClient) *BTCHeaderGenerator {
	return &BTCHeaderGenerator{
		t:      t,
		tm:     tm,
		client: client,
		wg:     &sync.WaitGroup{},
		quit:   make(chan struct{}),
	}
}

func (s *BTCHeaderGenerator) CatchUpBTCLightClient() {
	btcHeight, err := s.tm.TestRpcClient.GetBlockCount()
	require.NoError(s.t, err)

	tipResp, err := s.client.BTCHeaderChainTip()
	require.NoError(s.t, err)
	btclcHeight := tipResp.Header.Height

	var headers []*wire.BlockHeader
	for i := int(btclcHeight + 1); i <= int(btcHeight); i++ {
		hash, err := s.tm.TestRpcClient.GetBlockHash(int64(i))
		require.NoError(s.t, err)
		header, err := s.tm.TestRpcClient.GetBlockHeader(hash)
		require.NoError(s.t, err)
		headers = append(headers, header)
	}

	_, err = s.client.InsertBTCHeadersToBabylon(headers)
	require.NoError(s.t, err)
}

func (g *BTCHeaderGenerator) Start() {
	g.CatchUpBTCLightClient()
	g.wg.Add(1)
	go g.loop()
}

func (g *BTCHeaderGenerator) Stop() {
	close(g.quit)
	g.wg.Wait()
}

func (g *BTCHeaderGenerator) loop() {
	defer g.wg.Done()

	t := time.NewTicker(5 * time.Second)

	for {
		select {
		case <-g.quit:
			return
		case <-t.C:
			resp := g.tm.BitcoindHandler.GenerateBlocks(1)
			hash, err := chainhash.NewHashFromStr(resp.Blocks[0])
			require.NoError(g.t, err)
			block, err := g.tm.TestRpcClient.GetBlock(hash)
			require.NoError(g.t, err)
			_, err = g.client.InsertBTCHeadersToBabylon([]*wire.BlockHeader{&block.Header})
			require.NoError(g.t, err)

			btcHeight, err := g.tm.TestRpcClient.GetBlockCount()
			require.NoError(g.t, err)

			tipResp, err := g.client.BTCHeaderChainTip()
			require.NoError(g.t, err)
			btclcHeight := tipResp.Header.Height

			fmt.Printf("Current best block height: %d, BTC light client height: %d\n", btcHeight, btclcHeight)
		}
	}
}

type BTCStaker struct {
	t      *testing.T
	tm     *TestManager
	client *SenderWithBabylonClient
	fpPK   *btcec.PublicKey
	wg     *sync.WaitGroup
	quit   chan struct{}
}

func NewBTCStaker(
	t *testing.T,
	tm *TestManager,
	client *SenderWithBabylonClient,
	finalityProviderPublicKey *btcec.PublicKey,
) *BTCStaker {
	return &BTCStaker{
		t:      t,
		tm:     tm,
		client: client,
		fpPK:   finalityProviderPublicKey,
		wg:     &sync.WaitGroup{},
		quit:   make(chan struct{}),
	}
}

func (s *BTCStaker) Start() {
	stakerAddres, err := s.tm.TestRpcClient.GetNewAddress("")
	require.NoError(s.t, err)
	stakerInfo, err := s.tm.TestRpcClient.GetAddressInfo(stakerAddres.String())
	require.NoError(s.t, err)

	stakerPubKey, err := hex.DecodeString(*stakerInfo.PubKey)
	require.NoError(s.t, err)
	pk, err := btcec.ParsePubKey(stakerPubKey)
	require.NoError(s.t, err)

	s.wg.Add(1)
	go s.loop(stakerAddres, pk)
}

func (s *BTCStaker) Stop() {
	close(s.quit)
	s.wg.Wait()
}

// infinite loop to constantly send delegations
func (s *BTCStaker) loop(stakerAddres btcutil.Address, stakerPk *btcec.PublicKey) {
	defer s.wg.Done()

	for {
		select {
		case <-s.quit:
			return
		default:
			paramsResp, err := s.client.BTCStakingParams()
			require.NoError(s.t, err)
			s.buildAndSendStakingTransaction(stakerAddres, stakerPk, &paramsResp.Params)
		}
	}
}

func (s *BTCStaker) NewBabylonBip322Pop(
	msg []byte,
	w wire.TxWitness,
	a btcutil.Address) *btcstypes.ProofOfPossessionBTC {
	err := bip322.Verify(msg, w, a, nil)
	require.NoError(s.t, err)
	serializedWitness, err := bip322.SerializeWitness(w)
	require.NoError(s.t, err)
	bip322Sig := btcstypes.BIP322Sig{
		Sig:     serializedWitness,
		Address: a.EncodeAddress(),
	}
	m, err := bip322Sig.Marshal()
	require.NoError(s.t, err)
	pop := &btcstypes.ProofOfPossessionBTC{
		BtcSigType: btcstypes.BTCSigType(btcstypes.BTCSigType_BIP322),
		BtcSig:     m,
	}
	return pop
}

func (s *BTCStaker) signBip322NativeSegwit(stakerAddress btcutil.Address) (*btcstypes.ProofOfPossessionBTC, error) {
	babylonAddrHash := tmhash.Sum(s.client.BabylonAddress.Bytes())

	toSpend, err := bip322.GetToSpendTx(babylonAddrHash, stakerAddress)

	if err != nil {
		return nil, fmt.Errorf("failed to bip322 to spend tx: %w", err)
	}

	if !txscript.IsPayToWitnessPubKeyHash(toSpend.TxOut[0].PkScript) {
		return nil, fmt.Errorf("Bip322NativeSegwit support only native segwit addresses")
	}

	toSpendhash := toSpend.TxHash()

	toSign := bip322.GetToSignTx(toSpend)

	amt := float64(0)
	signed, all, err := s.tm.TestRpcClient.SignRawTransactionWithWallet2(toSign, []btcjson.RawTxWitnessInput{
		{
			Txid:         toSpendhash.String(),
			Vout:         0,
			ScriptPubKey: hex.EncodeToString(toSpend.TxOut[0].PkScript),
			Amount:       &amt,
		},
	})

	if err != nil {
		return nil, fmt.Errorf("failed to sign raw transaction while creating bip322 signature: %w", err)
	}

	if !all {
		return nil, fmt.Errorf("failed to create bip322 signature")
	}

	return s.NewBabylonBip322Pop(
		babylonAddrHash,
		signed.TxIn[0].Witness,
		stakerAddress,
	), nil
}

type SpendPathDescription struct {
	ControlBlock *txscript.ControlBlock
	ScriptLeaf   *txscript.TapLeaf
}

type TaprootSigningRequest struct {
	FundingOutput    *wire.TxOut
	TxToSign         *wire.MsgTx
	SpendDescription *SpendPathDescription
}

// TaprootSigningResult contains result of signing taproot spend through bitcoind
// wallet. It will contain either Signature or FullInputWitness, never both.
type TaprootSigningResult struct {
	Signature        *schnorr.Signature
	FullInputWitness wire.TxWitness
}

func (s *BTCStaker) SignOneInputTaprootSpendingTransaction(
	stakerPubKey *btcec.PublicKey,
	request *TaprootSigningRequest,
) (*TaprootSigningResult, error) {
	if len(request.TxToSign.TxIn) != 1 {
		return nil, fmt.Errorf("cannot sign transaction with more than one input")
	}

	if !txscript.IsPayToTaproot(request.FundingOutput.PkScript) {
		return nil, fmt.Errorf("cannot sign transaction spending non-taproot output")
	}

	psbtPacket, err := psbt.New(
		[]*wire.OutPoint{&request.TxToSign.TxIn[0].PreviousOutPoint},
		request.TxToSign.TxOut,
		request.TxToSign.Version,
		request.TxToSign.LockTime,
		[]uint32{request.TxToSign.TxIn[0].Sequence},
	)

	if err != nil {
		return nil, fmt.Errorf("failed to create PSBT packet with transaction to sign: %w", err)
	}

	psbtPacket.Inputs[0].SighashType = txscript.SigHashDefault
	psbtPacket.Inputs[0].WitnessUtxo = request.FundingOutput
	psbtPacket.Inputs[0].Bip32Derivation = []*psbt.Bip32Derivation{
		{
			PubKey: stakerPubKey.SerializeCompressed(),
		},
	}

	ctrlBlockBytes, err := request.SpendDescription.ControlBlock.ToBytes()

	if err != nil {
		return nil, fmt.Errorf("failed to serialize control block: %w", err)
	}

	psbtPacket.Inputs[0].TaprootLeafScript = []*psbt.TaprootTapLeafScript{
		{
			ControlBlock: ctrlBlockBytes,
			Script:       request.SpendDescription.ScriptLeaf.Script,
			LeafVersion:  request.SpendDescription.ScriptLeaf.LeafVersion,
		},
	}

	psbtEncoded, err := psbtPacket.B64Encode()

	if err != nil {
		return nil, fmt.Errorf("failed to encode PSBT packet: %w", err)
	}

	sign := true
	signResult, err := s.tm.TestRpcClient.WalletProcessPsbt(
		psbtEncoded,
		&sign,
		"DEFAULT",
		nil,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to sign PSBT packet: %w", err)
	}

	decodedBytes, err := base64.StdEncoding.DecodeString(signResult.Psbt)

	if err != nil {
		return nil, fmt.Errorf("failed to decode signed PSBT packet from b64: %w", err)
	}

	decodedPsbt, err := psbt.NewFromRawBytes(bytes.NewReader(decodedBytes), false)

	if err != nil {
		return nil, fmt.Errorf("failed to decode signed PSBT packet from bytes: %w", err)
	}

	// In our signing request we only handle transaction with one input, and request
	// signature for one public key, thus we can receive at most one signature from btc
	if len(decodedPsbt.Inputs[0].TaprootScriptSpendSig) == 1 {
		schnorSignature := decodedPsbt.Inputs[0].TaprootScriptSpendSig[0].Signature

		parsedSignature, err := schnorr.ParseSignature(schnorSignature)

		if err != nil {
			return nil, fmt.Errorf("failed to parse schnorr signature in psbt packet: %w", err)
		}

		return &TaprootSigningResult{
			Signature: parsedSignature,
		}, nil
	}

	// decodedPsbt.Inputs[0].TaprootScriptSpendSig was 0, it is possible that script
	// required only one signature to build whole witness
	if len(decodedPsbt.Inputs[0].FinalScriptWitness) > 0 {
		// we go whole witness, return it to the caller
		witness, err := bip322.SimpleSigToWitness(decodedPsbt.Inputs[0].FinalScriptWitness)

		if err != nil {
			return nil, fmt.Errorf("failed to parse witness in psbt packet: %w", err)
		}

		return &TaprootSigningResult{
			FullInputWitness: witness,
		}, nil
	}

	// neither witness, nor signature is filled.
	return nil, fmt.Errorf("no signature found in PSBT packet. Wallet can't sign given tx")
}

func GenerateProof(block *wire.MsgBlock, txIdx uint32) ([]byte, error) {

	headerBytes := bbntypes.NewBTCHeaderBytesFromBlockHeader(&block.Header)

	var txsBytes [][]byte
	for _, tx := range block.Transactions {
		bytes, err := bbntypes.SerializeBTCTx(tx)

		if err != nil {
			return nil, err
		}

		txsBytes = append(txsBytes, bytes)
	}

	proof, err := btcctypes.SpvProofFromHeaderAndTransactions(&headerBytes, txsBytes, uint(txIdx))

	if err != nil {
		return nil, err
	}

	return proof.MerkleNodes, nil
}

func (s *BTCStaker) waitForTransactionConfirmation(
	txHash *chainhash.Hash,
	requiredDepth uint32,
) *bstypes.InclusionProof {

	t := time.NewTicker(10 * time.Second)

	for {
		select {
		case <-t.C:
			tx, err := s.tm.TestRpcClient.GetTransaction(txHash)
			require.NoError(s.t, err)
			// add + 1 to be sure babylon light client is updated to correct height
			if tx.Confirmations > int64(requiredDepth)+1 {
				blockHash, err := chainhash.NewHashFromStr(tx.BlockHash)
				require.NoError(s.t, err)
				block, err := s.tm.TestRpcClient.GetBlock(blockHash)
				require.NoError(s.t, err)
				proof, err := GenerateProof(block, uint32(tx.BlockIndex))
				require.NoError(s.t, err)

				headerHAsh := bbntypes.NewBTCHeaderHashBytesFromChainhash(blockHash)

				return &bstypes.InclusionProof{
					Key: &btckpttypes.TransactionKey{
						Hash:  &headerHAsh,
						Index: uint32(tx.BlockIndex),
					},
					Proof: proof,
				}
			}
		case <-s.quit:
			return nil
		}

	}
}

func (s *BTCStaker) buildAndSendStakingTransaction(
	stakerAddres btcutil.Address,
	stakerPk *btcec.PublicKey,
	params *btcstypes.Params,
) {
	unbondingTime := uint16(100)
	covKeys, err := bbnPksToBtcPks(params.CovenantPks)
	require.NoError(s.t, err)

	stakingInfo, err := staking.BuildStakingInfo(
		stakerPk,
		[]*btcec.PublicKey{s.fpPK},
		covKeys,
		params.CovenantQuorum,
		uint16(params.MaxStakingTimeBlocks),
		btcutil.Amount(params.MinStakingValueSat),
		regtestParams,
	)
	require.NoError(s.t, err)

	stakingTx, hash := s.tm.AtomicFundSignSendStakingTx(s.t, stakingInfo.StakingOutput)
	fmt.Printf("send staking tx with hash %s \n", hash)

	// TODO: hardcoded two in tests
	inclusionProof := s.waitForTransactionConfirmation(hash, 2)

	if inclusionProof == nil {
		// we are quiting
		return
	}

	fmt.Printf("staking tx confirmed with hash %s \n", hash)

	unbondingTxValue := params.MinStakingValueSat - params.UnbondingFeeSat

	serializedStakingTx, err := bbntypes.SerializeBTCTx(stakingTx)
	require.NoError(s.t, err)

	unbondingTx := wire.NewMsgTx(2)

	unbondingTx.AddTxIn(wire.NewTxIn(
		wire.NewOutPoint(hash, 0),
		nil,
		nil,
	))
	unbondingInfo, err := staking.BuildUnbondingInfo(
		stakerPk,
		[]*btcec.PublicKey{s.fpPK},
		covKeys,
		params.CovenantQuorum,
		unbondingTime,
		btcutil.Amount(unbondingTxValue),
		regtestParams,
	)

	require.NoError(s.t, err)
	unbondingTx.AddTxOut(unbondingInfo.UnbondingOutput)

	serializedUnbondingTx, err := bbntypes.SerializeBTCTx(unbondingTx)
	require.NoError(s.t, err)

	// build slashing for staking and unbondidn
	stakingSlashing, err := staking.BuildSlashingTxFromStakingTxStrict(
		stakingTx,
		0,
		params.SlashingPkScript,
		stakerPk,
		unbondingTime,
		params.UnbondingFeeSat,
		params.SlashingRate,
		regtestParams,
	)
	require.NoError(s.t, err)

	stakingSlashingPath, err := stakingInfo.SlashingPathSpendInfo()
	require.NoError(s.t, err)

	unbondingSlashing, err := staking.BuildSlashingTxFromStakingTxStrict(
		unbondingTx,
		0,
		params.SlashingPkScript,
		stakerPk,
		unbondingTime,
		params.UnbondingFeeSat,
		params.SlashingRate,
		regtestParams,
	)
	require.NoError(s.t, err)
	unbondingSlashingPath, err := unbondingInfo.SlashingPathSpendInfo()
	require.NoError(s.t, err)

	signStakingSlashingRes, err := s.SignOneInputTaprootSpendingTransaction(stakerPk, &TaprootSigningRequest{
		FundingOutput: stakingTx.TxOut[0],
		TxToSign:      stakingSlashing,
		SpendDescription: &SpendPathDescription{
			ControlBlock: &stakingSlashingPath.ControlBlock,
			ScriptLeaf:   &stakingSlashingPath.RevealedLeaf,
		},
	})
	require.NoError(s.t, err)

	signUnbondingSlashingRes, err := s.SignOneInputTaprootSpendingTransaction(stakerPk, &TaprootSigningRequest{
		FundingOutput: unbondingTx.TxOut[0],
		TxToSign:      unbondingSlashing,
		SpendDescription: &SpendPathDescription{
			ControlBlock: &unbondingSlashingPath.ControlBlock,
			ScriptLeaf:   &unbondingSlashingPath.RevealedLeaf,
		},
	})
	require.NoError(s.t, err)

	stakingSlashingTx, err := bbntypes.SerializeBTCTx(stakingSlashing)
	require.NoError(s.t, err)
	stakingSlashingSig := bbntypes.NewBIP340SignatureFromBTCSig(signStakingSlashingRes.Signature)
	unbondingSlashingTx, err := bbntypes.SerializeBTCTx(unbondingSlashing)
	require.NoError(s.t, err)
	unbondingSlashingSig := bbntypes.NewBIP340SignatureFromBTCSig(signUnbondingSlashingRes.Signature)

	pop, err := s.signBip322NativeSegwit(stakerAddres)
	require.NoError(s.t, err)

	msgBTCDel := &bstypes.MsgCreateBTCDelegation{
		StakerAddr:              s.client.BabylonAddress.String(),
		Pop:                     pop,
		BtcPk:                   bbntypes.NewBIP340PubKeyFromBTCPK(stakerPk),
		FpBtcPkList:             []bbntypes.BIP340PubKey{*bbntypes.NewBIP340PubKeyFromBTCPK(s.fpPK)},
		StakingTime:             params.MaxStakingTimeBlocks,
		StakingValue:            params.MinStakingValueSat,
		StakingTx:               serializedStakingTx,
		StakingTxInclusionProof: inclusionProof,
		SlashingTx:              bstypes.NewBtcSlashingTxFromBytes(stakingSlashingTx),
		DelegatorSlashingSig:    stakingSlashingSig,
		// Unbonding related data
		UnbondingTime:                 uint32(unbondingTime),
		UnbondingTx:                   serializedUnbondingTx,
		UnbondingValue:                unbondingTxValue,
		UnbondingSlashingTx:           bstypes.NewBtcSlashingTxFromBytes(unbondingSlashingTx),
		DelegatorUnbondingSlashingSig: unbondingSlashingSig,
	}

	resp, err := s.client.SendMsgs(context.Background(), []sdk.Msg{msgBTCDel})
	require.NoError(s.t, err)
	require.NotNil(s.t, resp)
	fmt.Printf("Delegation sent for transaction with hash %s\n", hash)
}

// TODO: Add
// - finlity providers

type CovenanEmulator struct {
	t      *testing.T
	tm     *TestManager
	client *SenderWithBabylonClient
	covKey *btcec.PrivateKey
	wg     *sync.WaitGroup
	quit   chan struct{}
}

func NewCovenantEmulator(
	t *testing.T,
	tm *TestManager,
	covKey *btcec.PrivateKey,
	client *SenderWithBabylonClient,
) *CovenanEmulator {
	return &CovenanEmulator{
		t:      t,
		tm:     tm,
		client: client,
		covKey: covKey,
		wg:     &sync.WaitGroup{},
		quit:   make(chan struct{}),
	}
}

func (c *CovenanEmulator) Start() {
	c.wg.Add(1)
	go c.loop()
}

func (c *CovenanEmulator) Stop() {
	close(c.quit)
	c.wg.Wait()
}

func (c *CovenanEmulator) loop() {
	defer c.wg.Done()

	ticker := time.NewTicker(10 * time.Second)

	for {
		select {
		case <-c.quit:
			return
		case <-ticker.C:
			params, err := c.client.BTCStakingParams()
			require.NoError(c.t, err)
			respo, err := c.client.BTCDelegations(bstypes.BTCDelegationStatus_PENDING, nil)
			require.NoError(c.t, err)
			require.NotNil(c.t, respo)

			if len(respo.BtcDelegations) == 0 {
				continue
			}

			messages := c.getMessagesWithSIgnatures(respo.BtcDelegations, &params.Params)

			resp, err := c.client.SendMsgs(context.Background(), messages)
			require.NoError(c.t, err)
			require.NotNil(c.t, resp)
			fmt.Printf("sent %d covenant messages for delegations\n", len(messages))
		}
	}
}

// SignTransactions receives BTC delegation transactions to sign and returns all the signatures needed if nothing fails.

func (c *CovenanEmulator) covenantSignatures(
	fpEncKey *asig.EncryptionKey,
	stakingSlashingTx *wire.MsgTx,
	stakingTx *wire.MsgTx,
	stakingOutputIdx uint32,
	stakingUnbondingScritpt []byte,
	stakingSlashingScript []byte,
	unbondingTx *wire.MsgTx,
	slashUnbondingTx *wire.MsgTx,
	unbondingSlashingScript []byte,
) (slashSig, slashUnbondingSig *asig.AdaptorSignature, unbondingSig *schnorr.Signature) {
	// creates slash sigs
	slashSig, err := staking.EncSignTxWithOneScriptSpendInputStrict(
		stakingSlashingTx,
		stakingTx,
		stakingOutputIdx,
		stakingSlashingScript,
		c.covKey,
		fpEncKey,
	)
	require.NoError(c.t, err)
	// creates slash unbonding sig
	slashUnbondingSig, err = staking.EncSignTxWithOneScriptSpendInputStrict(
		slashUnbondingTx,
		unbondingTx,
		0, // 0th output is always the unbonding script output
		unbondingSlashingScript,
		c.covKey,
		fpEncKey,
	)
	require.NoError(c.t, err)

	unbondingSig, err = staking.SignTxWithOneScriptSpendInputStrict(
		unbondingTx,
		stakingTx,
		stakingOutputIdx,
		stakingUnbondingScritpt,
		c.covKey,
	)
	require.NoError(c.t, err)
	return slashSig, slashUnbondingSig, unbondingSig
}

func (c *CovenanEmulator) getMessagesWithSIgnatures(resp []*bstypes.BTCDelegationResponse, params *bstypes.Params) []sdk.Msg {
	var msgs []sdk.Msg

	for _, del := range resp {
		stakingTx, _, err := bbntypes.NewBTCTxFromHex(del.StakingTxHex)
		require.NoError(c.t, err)
		stakingSlashingTx, _, err := bbntypes.NewBTCTxFromHex(del.SlashingTxHex)
		require.NoError(c.t, err)
		unbondingTx, _, err := bbntypes.NewBTCTxFromHex(del.UndelegationResponse.UnbondingTxHex)
		require.NoError(c.t, err)
		slashUnbondingTx, _, err := bbntypes.NewBTCTxFromHex(del.UndelegationResponse.SlashingTxHex)
		require.NoError(c.t, err)
		stakerPk := del.BtcPk.MustToBTCPK()
		fpPk := del.FpBtcPkList[0].MustToBTCPK()
		stakingTime := del.EndHeight - del.StartHeight
		covenatKeys, err := bbnPksToBtcPks(params.CovenantPks)
		require.NoError(c.t, err)

		fpEncKey, err := asig.NewEncryptionKeyFromBTCPK(fpPk)
		require.NoError(c.t, err)

		stakingInfo, err := staking.BuildStakingInfo(
			stakerPk,
			[]*btcec.PublicKey{fpPk},
			covenatKeys,
			params.CovenantQuorum,
			uint16(stakingTime),
			btcutil.Amount(del.TotalSat),
			regtestParams,
		)
		require.NoError(c.t, err)

		unbondingInfo, err := staking.BuildUnbondingInfo(
			stakerPk,
			[]*btcec.PublicKey{fpPk},
			covenatKeys,
			params.CovenantQuorum,
			uint16(del.UnbondingTime),
			btcutil.Amount(unbondingTx.TxOut[0].Value),
			regtestParams,
		)
		require.NoError(c.t, err)

		stakingSlahingPath, err := stakingInfo.SlashingPathSpendInfo()
		require.NoError(c.t, err)
		stakingUnbondingPath, err := stakingInfo.UnbondingPathSpendInfo()
		require.NoError(c.t, err)
		unbondingSlashingPath, err := unbondingInfo.SlashingPathSpendInfo()
		require.NoError(c.t, err)
		stakingSlashingSig, unbondingSlashingSig, unbondingSig := c.covenantSignatures(
			fpEncKey,
			stakingSlashingTx,
			stakingTx,
			del.StakingOutputIdx,
			stakingUnbondingPath.RevealedLeaf.Script,
			stakingSlahingPath.RevealedLeaf.Script,
			unbondingTx,
			slashUnbondingTx,
			unbondingSlashingPath.RevealedLeaf.Script,
		)

		stakingTxHash := stakingTx.TxHash()
		msg := &bstypes.MsgAddCovenantSigs{
			Signer:                  c.client.BabylonAddress.String(),
			Pk:                      bbntypes.NewBIP340PubKeyFromBTCPK(c.covKey.PubKey()),
			StakingTxHash:           stakingTxHash.String(),
			SlashingTxSigs:          [][]byte{stakingSlashingSig.MustMarshal()},
			UnbondingTxSig:          bbntypes.NewBIP340SignatureFromBTCSig(unbondingSig),
			SlashingUnbondingTxSigs: [][]byte{unbondingSlashingSig.MustMarshal()},
		}

		msgs = append(msgs, msg)
	}

	return msgs
}

type SubReporter struct {
	t      *testing.T
	tm     *TestManager
	client *SenderWithBabylonClient
	wg     *sync.WaitGroup
	quit   chan struct{}
}

func NewSubReporter(
	t *testing.T,
	tm *TestManager,
	client *SenderWithBabylonClient,
) *SubReporter {
	return &SubReporter{
		t:      t,
		tm:     tm,
		client: client,
		wg:     &sync.WaitGroup{},
		quit:   make(chan struct{}),
	}
}

func (s *SubReporter) Start() {
	s.wg.Add(1)
	go s.loop()
}

func (s *SubReporter) Stop() {
	close(s.quit)
	s.wg.Wait()
}

func (s *SubReporter) loop() {
	defer s.wg.Done()

	ticker := time.NewTicker(10 * time.Second)

	for {
		select {
		case <-s.quit:
			return
		case <-ticker.C:
			// last ea
			resp, err := s.client.RawCheckpointList(checkpointingtypes.Sealed, nil)

			if err != nil {
				fmt.Printf("failed to get checkpoints %s\n", err)
				continue
			}

			if len(resp.RawCheckpoints) == 0 {
				continue
			}

			firstSelead := resp.RawCheckpoints[0]

			fmt.Printf("retrieved checkpoint for epoch %d\n", firstSelead.Ckpt.EpochNum)

			s.BuildSendReporCheckpoint(firstSelead.Ckpt)
		}
	}
}

func (s *SubReporter) encodeCheckpointData(ckpt *checkpointingtypes.RawCheckpointResponse) ([]byte, []byte, error) {
	// Convert to raw checkpoint
	rawCkpt, err := ckpt.ToRawCheckpoint()
	if err != nil {
		return nil, nil, err
	}

	// Convert raw checkpoint to BTC checkpoint
	btcCkpt, err := checkpointingtypes.FromRawCkptToBTCCkpt(rawCkpt, s.client.BabylonAddress)
	if err != nil {
		return nil, nil, err
	}

	// Encode checkpoint data
	data1, data2, err := btctxformatter.EncodeCheckpointData(
		babylonTag,
		0,
		btcCkpt,
	)
	if err != nil {
		return nil, nil, err
	}

	// Return the encoded data
	return data1, data2, nil
}

func (s *SubReporter) BuildSendReporCheckpoint(ckpt *checkpointingtypes.RawCheckpointResponse) {
	data1, data2, err := s.encodeCheckpointData(ckpt)
	require.NoError(s.t, err)

	builder1 := txscript.NewScriptBuilder()
	dataScript1, err := builder1.AddOp(txscript.OP_RETURN).AddData(data1).Script()
	require.NoError(s.t, err)

	builder2 := txscript.NewScriptBuilder()
	dataScript2, err := builder2.AddOp(txscript.OP_RETURN).AddData(data2).Script()
	require.NoError(s.t, err)

	dataTx1 := wire.NewMsgTx(2)
	dataTx1.AddTxOut(wire.NewTxOut(0, dataScript1))

	dataTx2 := wire.NewMsgTx(2)
	dataTx2.AddTxOut(wire.NewTxOut(0, dataScript2))

	_, hash1 := s.tm.AtomicFundSignSendStakingTx(s.t, wire.NewTxOut(0, dataScript1))
	_, hash2 := s.tm.AtomicFundSignSendStakingTx(s.t, wire.NewTxOut(0, dataScript2))

	proofs := s.waitFor2TransactionsConfirmation(hash1, hash2, 2)

	if len(proofs) == 0 {
		// we are quiting
		return
	}

	fmt.Printf("sending checkpoint for epoch %d with proof %d \n", ckpt.EpochNum, len(proofs))

	msg := &btckpttypes.MsgInsertBTCSpvProof{
		Submitter: s.client.BabylonAddress.String(),
		Proofs:    proofs,
	}

	resp, err := s.client.SendMsgs(context.Background(), []sdk.Msg{msg})
	require.NoError(s.t, err)
	require.NotNil(s.t, resp)
}

func (s *SubReporter) waitFor2TransactionsConfirmation(
	txHash *chainhash.Hash,
	txHash2 *chainhash.Hash,
	requiredDepth uint32,
) []*btckpttypes.BTCSpvProof {

	t := time.NewTicker(10 * time.Second)

	for {
		select {
		case <-t.C:
			tx1, err := s.tm.TestRpcClient.GetTransaction(txHash)
			require.NoError(s.t, err)
			tx2, err := s.tm.TestRpcClient.GetTransaction(txHash2)
			require.NoError(s.t, err)

			if tx1.Confirmations > int64(requiredDepth)+1 && tx2.Confirmations > int64(requiredDepth)+1 {
				blockHash, err := chainhash.NewHashFromStr(tx1.BlockHash)
				require.NoError(s.t, err)
				block1, err := s.tm.TestRpcClient.GetBlock(blockHash)
				require.NoError(s.t, err)
				proof, err := GenerateProof(block1, uint32(tx1.BlockIndex))
				require.NoError(s.t, err)
				tx1Bytes, err := hex.DecodeString(tx1.Hex)
				require.NoError(s.t, err)
				block1Header := bbntypes.NewBTCHeaderBytesFromBlockHeader(&block1.Header)

				blockHash2, err := chainhash.NewHashFromStr(tx2.BlockHash)
				require.NoError(s.t, err)
				block2, err := s.tm.TestRpcClient.GetBlock(blockHash2)
				require.NoError(s.t, err)
				blco2Header := bbntypes.NewBTCHeaderBytesFromBlockHeader(&block2.Header)
				proof2, err := GenerateProof(block2, uint32(tx2.BlockIndex))
				require.NoError(s.t, err)
				tx2Bytes, err := hex.DecodeString(tx2.Hex)
				require.NoError(s.t, err)

				return []*btckpttypes.BTCSpvProof{
					{
						BtcTransaction:      tx1Bytes,
						BtcTransactionIndex: uint32(tx1.BlockIndex),
						MerkleNodes:         proof,
						ConfirmingBtcHeader: &block1Header,
					},
					{
						BtcTransaction:      tx2Bytes,
						BtcTransactionIndex: uint32(tx2.BlockIndex),
						MerkleNodes:         proof2,
						ConfirmingBtcHeader: &blco2Header,
					},
				}
			}

		case <-s.quit:
			return nil
		}

	}
}
