package storage

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/parquet-go/parquet-go"
)

func TestWriterFlushesParquetFile(t *testing.T) {
	dir := t.TempDir()
	w := NewWriter(dir, 3)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := w.Start(ctx); err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	now := time.Now().UTC()
	for i := int64(0); i < 5; i++ {
		w.Append(Tick{
			InstrumentToken: 100 + i,
			Timestamp:       now.Add(time.Duration(i) * time.Second).UnixMilli(),
			LastPrice:       100 + float64(i),
			Volume:          1 + i,
		})
	}

	// Wait for the flusher to write the batch (flushEvery = 3 rows triggers
	// a flush after 3, then another on shutdown).
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if files, _ := filepath.Glob(filepath.Join(dir, "*.parquet")); len(files) > 0 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	files, err := filepath.Glob(filepath.Join(dir, "*.parquet"))
	if err != nil {
		t.Fatalf("glob error = %v", err)
	}
	if len(files) == 0 {
		t.Fatal("expected at least one parquet file")
	}

	// Read the file back and verify the schema/rows parse.
	f, err := os.Open(files[0])
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	reader := parquet.NewGenericReader[Tick](f)
	defer reader.Close()

	var got int64
	for {
		rows := make([]Tick, 3)
		n, err := reader.Read(rows)
		got += int64(n)
		if err != nil {
			break
		}
	}
	if got == 0 {
		t.Fatal("expected to read back at least one row")
	}
}
