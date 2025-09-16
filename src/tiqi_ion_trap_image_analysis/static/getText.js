let xmlHttp = new XMLHttpRequest();
xmlHttp.onload = (err) => {
    let jsonResponse = JSON.parse(xmlHttp.response);

    for (let key in jsonResponse) {
        document.getElementById(key).innerText = jsonResponse[key];
    }
    if (Number(document.getElementById("fit_avg").innerText) >= 20) {
    document.getElementById("response").innerText = "Calibration needed!"
    }
    else{
        document.getElementById("response").innerText = ""
}
};

setInterval(() => {
    xmlHttp.open( "GET", '/stream_text', true );
    xmlHttp.send( null );
}, 1000);
