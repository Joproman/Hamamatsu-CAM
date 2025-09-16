const image = document.getElementById('myCanvas');

function drawX(ctx, x, y, color) {
    ctx.beginPath();

    ctx.moveTo(x, 0);
    ctx.lineTo(x, image.height);

    ctx.moveTo(0, y);
    ctx.lineTo(image.width, y);
    ctx.lineWidth = 0.5;
    ctx.strokeStyle = color;
    ctx.stroke();
}

const ctx = image.getContext("2d");

function updateCrossHair(old_x, old_y, new_x, new_y, color, num) {
    const old_coord = {"x": old_x, "y": old_y};
    const new_coord = {"x": new_x, "y": new_y};
    const new_row = document.getElementById("row_" + num);
    new_row.innerHTML = `x: <input type="number" class="xvalue_${num}" placeholder=${new_x} style="width: 70px" value=${new_x}> ; 
                         y: <input type="number" class="yvalue_${num}" placeholder=${new_y} style="width: 70px" value=${new_y}>  
                         <button type="button" id="updateButton_${num}" onclick="onClickEvent(this)">Update</button> 
                         <button type="button" id="deleteButton_${num}" onclick="deleteCrossHair($(this).closest('p'), ${new_x}, ${new_y})">Delete</button>
                         <span style="display: none;" id="color_${num}">${color}<span/>`;
    $.ajax({
        url: "/cross_hair/update",
        type: "get",
        data: {
            "old_coord": old_coord,
            "new_coord": new_coord,
            "color": color
        },
        success: function(response) {
            drawX(ctx, response["html_x"], response["html_y"], color);
            },
        error: function(jqxhr, status, exception) {
            alert('Exception:', exception);
        }
    });
    setTimeout(function () {
        ctx.clearRect(0, 0, image.width, image.height);
        const img = $('<img />', {src : '/capture'});
        ctx.drawImage(img[0], 0, 0, 700, 700);
    }, 800)
}

function deleteCrossHair(element, x, y) {
    element.remove()
    const coord = {"x": x, "y": y};
    $.ajax({
        url: "/cross_hair/delete",
        type: "get",
        data: {"coords": coord},
        success: function(response) {
            },
        error: function(jqxhr, status, exception) {
            alert('Exception:', exception);
        }
    });
    setTimeout(function () {
        ctx.clearRect(0, 0, image.width, image.height);
        const img = $('<img />', {src : '/capture'});
        ctx.drawImage(img[0], 0, 0, 700, 700);
    }, 800)
}

function onClickEvent(element){
    let num = element.id.slice(-1);
    const new_x = $('.xvalue_' + num)[0].value;
    const new_y = $('.yvalue_' + num)[0].value;
    const old_x = $('.xvalue_' + num)[0].placeholder;
    const old_y = $('.yvalue_' + num)[0].placeholder;
    const color = document.getElementById("color_" + num).innerText
    updateCrossHair(old_x, old_y, new_x, new_y, color, num)
}

const rect = image.getBoundingClientRect();
image.addEventListener('click', function (event){
    const coord = {"x": event.clientX - rect.left, "y": event.clientY - rect.top};
    $.ajax({
        url: "/cross_hair/add",
        type: "get",
        data: {"coords": coord},
        success: function(response) {
            let num = response["num"];
            let color = `rgb(${response['color'][0]}, ${response['color'][1]}, ${response['color'][2]})`;
            drawX(ctx, coord.x, coord.y, color);
            const new_row = document.createElement('p');
            document.getElementById('crossHairContainer').appendChild(new_row);
            new_row.id ="row_" + num;
            new_row.innerHTML = `x: <input type="number" class="xvalue_${num}" placeholder=${response['x']} style="width: 70px" value=${response['x']}> ; 
                                 y: <input type="number" class="yvalue_${num}" placeholder=${response['y']} style="width: 70px" value=${response['y']}>  
                                 <button type="button" id="updateButton_${num}" onclick="onClickEvent(this)">Update</button> 
                                 <button type="button" id="deleteButton_${num}" onclick="deleteCrossHair($(this).closest('p'), ${response['x']}, ${response['y']})">Delete</button>
                                 <span style="display: none;" id="color_${num}">${color}<span/>`;
            },
        error: function(jqxhr, status, exception) {
            alert('Exception:', exception);
        }
    });
    coords = [];
})