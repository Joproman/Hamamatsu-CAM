function drawX(ctx, x, y) {
    ctx.beginPath();

    ctx.moveTo(x - 5, y - 5);
    ctx.lineTo(x + 5, y + 5);

    ctx.moveTo(x + 5, y - 5);
    ctx.lineTo(x - 5, y + 5);
    ctx.strokeStyle = 'red';
    ctx.stroke();
}

var coords = [];
var image = document.getElementById('myCanvas');
var ctx = image.getContext("2d");
var rect = image.getBoundingClientRect();
image.addEventListener('click', function (event){
    var coord = { "x": event.clientX - rect.left, "y": event.clientY - rect.top};
    coords.push(coord);
    drawX(ctx, coord.x, coord.y);
    if (coords.length === 2) {
        $.ajax({
            url: "/ROI/update",
            type: "get",
            data: {"coords": coords},
            success: function(response) {
                alert("ROI has been updated");
                window.location = '/';
            },
            error: function(jqxhr, status, exception) {
                alert('Exception:', exception);
            }
        });
        coords = [];
    }
})
