#!/bin/bash
# Monitor data collection progress and check if we have enough data for validation

PLAYBOOK="memory/domains/hot_pot/playbook.json"
MIN_TOTAL=10
MIN_HIGH=3

echo "Monitoring data collection for fidelity validation experiments..."
echo "Requirements:"
echo "  - At least $MIN_TOTAL total episodes"
echo "  - At least $MIN_HIGH HIGH reliability episodes"
echo ""

while true; do
    # Check if playbook exists
    if [ -f "$PLAYBOOK" ]; then
        # Count episodes
        TOTAL=$(cat "$PLAYBOOK" | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('observations', [])))" 2>/dev/null || echo "0")
        HIGH=$(cat "$PLAYBOOK" | python3 -c "import sys, json; data=json.load(sys.stdin); print(sum(1 for o in data.get('observations', []) if o.get('reliability')=='HIGH'))" 2>/dev/null || echo "0")

        # Display progress
        echo -ne "\r[$(date +%H:%M:%S)] Episodes: $TOTAL/$MIN_TOTAL | HIGH reliability: $HIGH/$MIN_HIGH"

        # Check if we have enough
        if [ "$TOTAL" -ge "$MIN_TOTAL" ] && [ "$HIGH" -ge "$MIN_HIGH" ]; then
            echo ""
            echo ""
            echo "✓ SUCCESS! We have enough data for validation experiments!"
            echo "  - Total episodes: $TOTAL"
            echo "  - HIGH reliability: $HIGH"
            echo ""
            echo "Ready to run validation:"
            echo "  python3 experiments/fidelity_validation.py --domain hot_pot --output results/fidelity_validation_results.json"
            exit 0
        fi
    else
        echo -ne "\r[$(date +%H:%M:%S)] Waiting for playbook to be created..."
    fi

    sleep 10
done
