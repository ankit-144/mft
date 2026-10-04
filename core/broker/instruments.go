package broker

import (
	"bytes"
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/mft/core/contracts"
)

// Column names in the Kite instrument dump.
const (
	colInstrumentToken = "instrument_token"
	colExchange        = "exchange"
	colTradingSymbol   = "tradingsymbol"
	colExpiry          = "expiry"
	colOptionType      = "option_type"
	colLotSize         = "lot_size"
)

// Instruments returns the instrument master for the symbols declared in broker config,
// resolved to their Kite instrument tokens.
func (k *Kite) Instruments(ctx context.Context) ([]contracts.Instrument, error) {
	if err := k.checkAuth(); err != nil {
		return nil, err
	}

	k.instMu.Lock()
	defer k.instMu.Unlock()

	if len(k.instrument) > 0 && time.Since(k.loadedAt) < k.instrumentTTL {
		return append([]contracts.Instrument(nil), k.instrument...), nil
	}

	wanted := k.wantedSymbols()
	if len(wanted) == 0 {
		return nil, nil
	}

	body, err := k.do(ctx, httpGet, k.endpoints.instrumentsURL(), nil, instrumentDumpMaxBytes)
	if err != nil {
		return nil, fmt.Errorf("kite: instrument dump: %w", err)
	}

	found, err := parseInstrumentDump(body, wanted)
	if err != nil {
		return nil, err
	}

	k.instrument = found
	k.loadedAt = time.Now()
	return append([]contracts.Instrument(nil), found...), nil
}

// wantedSymbols returns the deduplicated, upper-cased config watchlist.
func (k *Kite) wantedSymbols() []string {
	seen := make(map[string]struct{}, len(k.watchlist))
	out := make([]string, 0, len(k.watchlist))
	for _, s := range k.watchlist {
		sym := strings.ToUpper(strings.TrimSpace(s))
		if sym == "" {
			continue
		}
		if _, dup := seen[sym]; dup {
			continue
		}
		seen[sym] = struct{}{}
		out = append(out, sym)
	}
	return out
}

// Instrument returns the resolved instrument for a single symbol, serving it from the
// cached master when possible.
func (k *Kite) Instrument(ctx context.Context, symbol string) (contracts.Instrument, error) {
	all, err := k.Instruments(ctx)
	if err != nil {
		return contracts.Instrument{}, err
	}
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	for _, inst := range all {
		if inst.Symbol == sym {
			return inst, nil
		}
	}
	return contracts.Instrument{}, fmt.Errorf("kite: symbol %q: %w", symbol, ErrInstrumentNotFound)
}

// lookupToken resolves a symbol against an already-loaded instrument set.
func lookupToken(all []contracts.Instrument, symbol string) (contracts.Instrument, error) {
	sym := strings.ToUpper(strings.TrimSpace(symbol))
	for _, inst := range all {
		if inst.Symbol == sym {
			return inst, nil
		}
	}
	return contracts.Instrument{}, fmt.Errorf("kite: symbol %q: %w", symbol, ErrInstrumentNotFound)
}

// parseInstrumentDump maps the CSV instrument dump onto the configured symbols.
func parseInstrumentDump(body []byte, want []string) ([]contracts.Instrument, error) {
	r := csv.NewReader(bytes.NewReader(body))
	r.FieldsPerRecord = -1
	r.ReuseRecord = false

	header, err := r.Read()
	if err != nil {
		return nil, fmt.Errorf("kite: instrument dump: read header: %w: %w", ErrUnavailable, err)
	}
	col := make(map[string]int, len(header))
	for i, name := range header {
		col[strings.ToLower(strings.TrimSpace(name))] = i
	}
	for _, required := range []string{colInstrumentToken, colExchange, colTradingSymbol} {
		if _, ok := col[required]; !ok {
			return nil, fmt.Errorf("kite: instrument dump: missing %q column: %w", required, ErrUnavailable)
		}
	}

	target := make(map[string]struct{}, len(want))
	for _, s := range want {
		target[strings.ToUpper(strings.TrimSpace(s))] = struct{}{}
	}

	best := make(map[string]contracts.Instrument, len(want))

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("kite: instrument dump: %w: %w", ErrUnavailable, err)
		}
		if len(record) <= col[colTradingSymbol] {
			continue
		}
		symbol := strings.ToUpper(strings.TrimSpace(record[col[colTradingSymbol]]))
		if _, ok := target[symbol]; !ok {
			continue
		}
		if derivative(record, col) {
			continue
		}

		inst, err := instrumentFromRecord(record, col)
		if err != nil {
			continue
		}
		if existing, seen := best[symbol]; !seen || instrumentRank(inst.Exchange) < instrumentRank(existing.Exchange) {
			best[symbol] = inst
		}
	}

	var missing []string
	out := make([]contracts.Instrument, 0, len(want))
	for _, s := range want {
		sym := strings.ToUpper(strings.TrimSpace(s))
		inst, ok := best[sym]
		if !ok {
			missing = append(missing, sym)
			continue
		}
		out = append(out, inst)
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return nil, errNoToken(missing)
	}
	return out, nil
}

// derivative reports whether an instrument dump row is a future or option leg rather
// than a cash instrument.
func derivative(record []string, col map[string]int) bool {
	if i, ok := col[colOptionType]; ok && i < len(record) {
		if strings.TrimSpace(record[i]) != "" {
			return true
		}
	}
	if i, ok := col[colExpiry]; ok && i < len(record) {
		if strings.TrimSpace(record[i]) != "" {
			return true
		}
	}
	return false
}

// instrumentFromRecord converts one CSV row into a contracts.Instrument.
func instrumentFromRecord(record []string, col map[string]int) (contracts.Instrument, error) {
	cell := func(name string) string {
		i, ok := col[name]
		if !ok || i >= len(record) {
			return ""
		}
		return strings.TrimSpace(record[i])
	}

	token, err := strconv.ParseInt(cell(colInstrumentToken), 10, 64)
	if err != nil {
		return contracts.Instrument{}, fmt.Errorf("kite: instrument token %q: %w: %w", cell(colInstrumentToken), ErrInstrumentNotFound, err)
	}

	lotSize := 1
	if raw := cell(colLotSize); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			return contracts.Instrument{}, fmt.Errorf("kite: lot size %q: %w: %w", raw, ErrInstrumentNotFound, err)
		}
		if parsed > 0 {
			lotSize = parsed
		}
	}

	return contracts.Instrument{
		Token:    token,
		Symbol:   strings.ToUpper(cell(colTradingSymbol)),
		Exchange: strings.ToUpper(cell(colExchange)),
		LotSize:  lotSize,
	}, nil
}

// instrumentRank orders candidate rows for one symbol so that mapping is deterministic
// when a symbol is listed on more than one exchange.
func instrumentRank(exchange string) int {
	switch exchange {
	case "NSE":
		return 0
	case "BSE":
		return 1
	case "NFO", "CDS", "MCX":
		return 2
	default:
		return 3
	}
}
