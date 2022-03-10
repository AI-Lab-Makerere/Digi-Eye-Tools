$(document).ready(function() {

// $('#editparams').click(function () {
// 	if($('#editparams').text() == 'Edit Params') {
// 		$('#editparams').text('Update');
// 	} else {
// 		$('#editparams').text('Edit Params');
// 	}
// });

function updateMode(){
	$('#editparams').addClass("hidden");
	$('#submitbutton').removeClass("hidden");
	$('#numroots').attr('readonly',false);
	$('#rootbkgd').attr('readonly',false);
	
}

function readOnlyMode(){
	$('#submitbutton').addClass("hidden");
	$('#editparams').removeClass("hidden");
	$('#numroots').attr('readonly',true);
	$('#rootbkgd').attr('readonly',true);
}
$('#editparams').click(function () {
	updateMode();	
});

$('#submitbutton').click(function () {
	readOnlyMode();
});

});