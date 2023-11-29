async function submitMove(move="") {
    const url="/api/move"
    let response = await fetch(url, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        redirect: "follow",
        body: JSON.stringify({"playerMove": move})
    });
    results = await response.json();

    const playerMoveDisplay = document.querySelector("#playerMove");
    const computerMoveDisplay = document.querySelector("#computerMove");
    const resultDisplay = document.querySelector("#result");

    playerMoveDisplay.innerText = "You played: " + move.toLowerCase();
    computerMoveDisplay.innerText =
        "Computer played: " + results.computerMove.toLowerCase();
    switch(results.winner) {
        case "PLAYER":
            resultDisplay.innerText = "You win!";
            break;
        case "COMPUTER":
            resultDisplay.innerText = "You lose!";
            break;
        case "TIE":
            resultDisplay.innerText = "It's a tie!";
            break;
    }
};
