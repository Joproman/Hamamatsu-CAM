function change_status(status_class, name) {
    var current_status = $(status_class).text();
    $.ajax({
        url: `/get_${name}_toggled_status`,
        type: "get",
        data: {status: current_status},
        success: function(response) {
            $(status_class).html(response);
        },
        error: function(jqxhr, status, exception) {
            alert('Exception:', exception);
        }
    });
}