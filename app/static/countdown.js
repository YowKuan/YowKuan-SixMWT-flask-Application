

const startseconds = 20;
let time = startseconds *1;

const countdownEl = document.getElementById('countdowns');

setInterval(updatedown, 1000);

function updatedown() {
    console.log('count');
    let seconds = time

    countdownEl.innerHTML = `${seconds}`;

    time--;
    time = time < 0 ? 0 : time; 
}