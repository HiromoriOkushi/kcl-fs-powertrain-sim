#!binbash
# Script to run the Formula Student Engine Program

echo ========================================================
echo Formula Student Engine Program Runner
echo ========================================================

# Ensure script is executable
if [ ! -x run_engine_demo.py ]; then
    echo Making run_engine_demo.py executable...
    chmod +x run_engine_demo.py
fi

# Run the engine demo
echo Starting engine demonstration...
python run_engine_demo.py

# Check if script ran successfully
if [ $ -eq 0 ]; then
    echo ========================================================
    echo Engine demonstration completed successfully.
    echo You can find the output in the dataoutputengine directory.
    echo ========================================================
else
    echo ========================================================
    echo Engine demonstration encountered errors.
    echo Please check the output for more details.
    echo ========================================================
fi