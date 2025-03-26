# Set up the environment (if using Anaconda, uncomment the following line)
# & "C:\Users\YourUsername\anaconda3\Scripts\activate.ps1"; conda activate logevaluate

Set-Location evaluation

# Define the techniques to evaluate
$techniques = @("BERT")

foreach ($technique in $techniques) {
    Write-Output $technique
    python "${technique}_eval.py" -otc
    python "${technique}_eval.py" -full
}
