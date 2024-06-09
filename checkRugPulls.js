// checkRugPulls.js
const web3 = require('@solana/web3.js');

const connection = new web3.Connection(web3.clusterApiUrl('mainnet-beta'), 'confirmed');

async function getTransactionHistory(publicKey) {
    return await connection.getConfirmedSignaturesForAddress2(publicKey, { limit: 1000 });
}

async function analyzeTransactionVolumeAndFrequency(transactions, historicalData) {
    let score = 0;
    const recentVolume = transactions.reduce((acc, tx) => acc + (tx.lamports || 0), 0);
    const averageVolume = historicalData.averageVolume;
    const frequency = transactions.length;
    const averageFrequency = historicalData.averageFrequency;

    if (recentVolume > 2 * averageVolume) {
        console.log("Warning: High transaction volume detected");
        score += 20;
    } else {
        console.log("Cleared: Transaction volume within normal limits");
    }

    if (frequency > 2 * averageFrequency) {
        console.log("Warning: High transaction frequency detected");
        score += 20;
    } else {
        console.log("Cleared: Transaction frequency within normal limits");
    }

    return score;
}

async function contextualAnalysis(accountInfo, transactions) {
    let score = 0;
    const accountAgeThreshold = 2592000000; // 30 days in milliseconds
    const currentTime = new Date().getTime();
    const accountCreationTime = accountInfo.creationTime || currentTime; 

    if ((currentTime - accountCreationTime) < accountAgeThreshold) {
        console.log("Warning: New account with large transactions detected");
        score += 30;
    } else {
        console.log("Cleared: Account age is satisfactory");
    }

    const flaggedAddresses = ["Address1", "Address2"];
    let flaggedTransactionFound = false;
    transactions.forEach(tx => {
        if (flaggedAddresses.includes(tx.to)) {
            console.log("Warning: Transaction with flagged address detected");
            flaggedTransactionFound = true;
            score += 50;
        }
    });

    if (!flaggedTransactionFound) {
        console.log("Cleared: No transactions with flagged addresses");
    }

    return score;
}

async function smartContractInteractions(transactions) {
    let score = 0;
    let contractInteractionDetected = false;
    transactions.forEach(tx => {
        if (tx.invokesSmartContract) {
            console.log("Warning: Smart contract interaction detected");
            contractInteractionDetected = true;
            score += 40;
        }
    });

    if (!contractInteractionDetected) {
        console.log("Cleared: No suspicious smart contract interactions");
    }

    return score;
}

async function calculateRugPullScore(publicKey) {
    const publicKeyObj = new web3.PublicKey(publicKey);
    const transactions = await getTransactionHistory(publicKeyObj);
    const historicalData = { averageVolume: 5000000, averageFrequency: 30 };

    const volumeFrequencyScore = await analyzeTransactionVolumeAndFrequency(transactions, historicalData);
    const contextualScore = await contextualAnalysis({ creationTime: new Date().setMonth(new Date().getMonth() - 6) }, transactions);
    const contractScore = await smartContractInteractions(transactions);

    const totalScore = volumeFrequencyScore + contextualScore + contractScore;
    
    const riskLevel = totalScore > 100 ? 'High' :
                      totalScore > 50 ? 'Medium' : 'Low';
    
    console.log(`Total rug pull score for ${publicKey}: ${riskLevel}`);
}

// Example usage
calculateRugPullScore(process.argv[2]);  // Take publicKey as a command line argument