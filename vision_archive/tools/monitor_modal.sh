#!/bin/bash
# Modal Usage Monitoring Dashboard
# Quick script to monitor your Modal GPU usage, costs, and logs

APP_NAME="socialsense-ar-gpu"
PROFILE="aadikrishna04"

echo "======================================================================"
echo "   Modal GPU Usage Monitor - SocialSenseAR"
echo "======================================================================"
echo ""

# Check if Modal CLI is available
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Function to display menu
show_menu() {
    echo ""
    echo "📊 What would you like to check?"
    echo ""
    echo "  1) 📈 Real-time logs (live updates)"
    echo "  2) 📋 Recent logs (last 50 lines)"
    echo "  3) 💰 Usage & costs (billing dashboard)"
    echo "  4) 🔍 App status (running/stopped)"
    echo "  5) 📊 Container stats (GPU/CPU/memory)"
    echo "  6) 🛑 Stop the app (save costs)"
    echo "  7) 🚀 Restart/redeploy"
    echo "  8) 🌐 Open web dashboard"
    echo "  9) ❌ Exit"
    echo ""
    read -p "Enter choice [1-9]: " choice
    echo ""
}

# Function to get app logs (live)
live_logs() {
    echo "📈 Streaming live logs from Modal GPU..."
    echo "   Press Ctrl+C to stop"
    echo ""
    modal app logs $APP_NAME --follow
}

# Function to get recent logs
recent_logs() {
    echo "📋 Recent logs (last 50 lines):"
    echo ""
    modal app logs $APP_NAME 2>&1 | tail -50
    echo ""
    read -p "Press Enter to continue..."
}

# Function to check usage/costs
check_costs() {
    echo "💰 Opening Modal billing dashboard..."
    echo ""
    echo "   Web Dashboard: https://modal.com/settings/billing"
    echo "   Current Profile: $PROFILE"
    echo ""
    echo "💡 Tip: Look for 'socialsense-ar-gpu' in the usage breakdown"
    echo ""

    # Try to open browser
    if command -v open &> /dev/null; then
        open "https://modal.com/settings/billing"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "https://modal.com/settings/billing"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Function to check app status
check_status() {
    echo "🔍 Checking app status..."
    echo ""
    modal app list | grep -A 5 "$APP_NAME" || echo "App not found in list"
    echo ""
    echo "📊 Deployment info:"
    echo "   https://modal.com/apps/$PROFILE/main/deployed/$APP_NAME"
    echo ""
    read -p "Press Enter to continue..."
}

# Function to get container stats
container_stats() {
    echo "📊 Container statistics..."
    echo ""
    echo "⚠️  Note: Modal CLI doesn't expose detailed stats via command line."
    echo "    Use the web dashboard for detailed metrics:"
    echo ""
    echo "    https://modal.com/apps/$PROFILE/main/deployed/$APP_NAME"
    echo ""

    # Try to open browser
    if command -v open &> /dev/null; then
        open "https://modal.com/apps/$PROFILE/main/deployed/$APP_NAME"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "https://modal.com/apps/$PROFILE/main/deployed/$APP_NAME"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Function to stop app
stop_app() {
    echo "🛑 Stopping Modal app..."
    echo ""
    read -p "Are you sure? This will stop the GPU server. (y/n): " confirm
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
        modal app stop $APP_NAME
        echo ""
        echo "✅ App stopped. No more charges until you redeploy."
    else
        echo "❌ Cancelled"
    fi
    echo ""
    read -p "Press Enter to continue..."
}

# Function to redeploy
redeploy_app() {
    echo "🚀 Redeploying app..."
    echo ""
    read -p "This will redeploy the latest code. Continue? (y/n): " confirm
    if [[ $confirm == "y" || $confirm == "Y" ]]; then
        cd "$(dirname "$0")/.." || exit
        modal deploy modal_app.py
        echo ""
        echo "✅ Redeployed!"
    else
        echo "❌ Cancelled"
    fi
    echo ""
    read -p "Press Enter to continue..."
}

# Function to open web dashboard
open_dashboard() {
    echo "🌐 Opening Modal web dashboard..."
    echo ""

    URL="https://modal.com/apps/$PROFILE/main/deployed/$APP_NAME"

    if command -v open &> /dev/null; then
        open "$URL"
    elif command -v xdg-open &> /dev/null; then
        xdg-open "$URL"
    else
        echo "   $URL"
    fi

    echo ""
    read -p "Press Enter to continue..."
}

# Main loop
while true; do
    show_menu

    case $choice in
        1)
            live_logs
            ;;
        2)
            recent_logs
            ;;
        3)
            check_costs
            ;;
        4)
            check_status
            ;;
        5)
            container_stats
            ;;
        6)
            stop_app
            ;;
        7)
            redeploy_app
            ;;
        8)
            open_dashboard
            ;;
        9)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-9."
            sleep 2
            ;;
    esac
done
