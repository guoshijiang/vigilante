package config

import (
	"errors"

	"github.com/babylonlabs-io/vigilante/types"
)

const (
	DefaultCheckpointCacheMaxEntries = 100
	DefaultPollingIntervalSeconds    = 60   // in seconds
	DefaultResendIntervalSeconds     = 1800 // 30 minutes
	DefaultResubmitFeeMultiplier     = 1
)

// SubmitterConfig defines configuration for the gRPC-web server.
type SubmitterConfig struct {
	// NetParams defines the BTC network params, which should be mainnet|testnet|simnet|signet
	NetParams string `mapstructure:"netparams"`
	// BufferSize defines the number of raw checkpoints stored in the buffer
	BufferSize uint `mapstructure:"buffer-size"`
	// ResubmitFeeMultiplier is used to multiply the estimated bumped fee in resubmission
	ResubmitFeeMultiplier float64 `mapstructure:"resubmit-fee-multiplier"`
	// PollingIntervalSeconds defines the intervals (in seconds) between each polling of Babylon checkpoints
	PollingIntervalSeconds int64 `mapstructure:"polling-interval-seconds"`
	// ResendIntervalSeconds defines the time (in seconds) which the submitter awaits
	// before resubmitting checkpoints to BTC
	ResendIntervalSeconds uint `mapstructure:"resend-interval-seconds"`
	// DatabaseConfig stores last submitted txn
	DatabaseConfig *DBConfig `mapstructure:"dbconfig"`
}

func (cfg *SubmitterConfig) Validate() error {
	if _, ok := types.GetValidNetParams()[cfg.NetParams]; !ok {
		return errors.New("invalid net params")
	}

	if cfg.ResubmitFeeMultiplier < 1 {
		return errors.New("invalid resubmit-fee-multiplier, should not be less than 1")
	}

	if cfg.PollingIntervalSeconds < 0 {
		return errors.New("invalid polling-interval-seconds, should be positive")
	}

	return nil
}

func DefaultSubmitterConfig() SubmitterConfig {
	return SubmitterConfig{
		NetParams:              types.BtcSimnet.String(),
		BufferSize:             DefaultCheckpointCacheMaxEntries,
		ResubmitFeeMultiplier:  DefaultResubmitFeeMultiplier,
		PollingIntervalSeconds: DefaultPollingIntervalSeconds,
		ResendIntervalSeconds:  DefaultResendIntervalSeconds,
	}
}
