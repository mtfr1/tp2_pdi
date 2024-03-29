<!DOCTYPE html>
<html lang="pt-BR">

<head>

  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
  <meta name="description" content="">
  <meta name="author" content="">

  <title>Classificação de frutas</title>

  <!-- Custom fonts for this theme -->
  <link href="theme/vendor/fontawesome-free/css/all.min.css" rel="stylesheet" type="text/css">
  <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css">
  <link href="https://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic" rel="stylesheet" type="text/css">

  <!-- Theme CSS -->
  <link href="theme/css/freelancer.min.css" rel="stylesheet">
  <link href="theme/css/teste.css" rel="stylesheet">
  
</head>


<!--  Setando variaveis  -->
<?php
  if(isset($_GET['image'])){
    $image = $_GET['image'];
  }
  else{
    $image = "images/fruta1.png";
  }
  if(isset($_GET['result'])){
    $result = $_GET['result'];
  }
  else{
    $result = "Não foi possível classificar a imagem!";
  }
  if(isset($_GET['tecnica'])){
    $tecnica = $_GET['tecnica'];
  }
  else{
    $tecnica = "lbp";
  }
?>

<body id="page-top">

  <!-- Navigation -->
  <nav class="navbar navbar-expand-lg bg-secondary text-uppercase fixed-top" id="mainNav">
    <div class="container">
      <a class="navbar-brand js-scroll-trigger" href="#page-top">Processamento Digital de Imagens</a>
      <button class="navbar-toggler navbar-toggler-right text-uppercase font-weight-bold bg-primary text-white rounded" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
        Menu
        <i class="fas fa-bars"></i>
      </button>
      <div class="collapse navbar-collapse" id="navbarResponsive">
        <ul class="navbar-nav ml-auto">
          <!-- <li class="nav-item mx-0 mx-lg-1">
            <a class="nav-link py-3 px-0 px-lg-3 rounded js-scroll-trigger" href="index.html">Retornar</a>
          </li> -->
          <li class="nav-item mx-0 mx-lg-1">
            <a class="nav-link py-3 px-0 px-lg-3 rounded js-scroll-trigger" href="#grupo">Grupo</a>
          </li>
        </ul>
      </div>
    </div> 
  </nav>

  <!-- Masthead -->
  <header class="masthead bg-primary text-white text-center">
    <div class="container d-flex align-items-center flex-column">

      <?php 
        if($tecnica == "lbp"){
          echo "<h1 class='masthead-heading text-uppercase mb-5'>LBP classifica como:</h1>";
        }
        if($tecnica == "har"){
          echo "<h1 class='masthead-heading text-uppercase mb-5'>Haralick classifica como:</h1>";
        }
        if($tecnica == "hist"){
          echo "<h1 class='masthead-heading text-uppercase mb-5'>Histograma de Cores classifica como:</h1>";
        }
      ?>

      <!-- Masthead Avatar Image -->
      <?php
        echo "<img class='masthead-avatar mb-5 rcorners2' src=".$image." alt=''>";
        
        echo "<h1 class='masthead-heading text-uppercase mb-0'>".$result."</h1>";
        
      ?>
        <!-- Icon Divider -->
      <div class="divider-custom divider-light">
        <div class="divider-custom-line"></div>
        <div class="divider-custom-icon">
          <i class="fas fa-star"></i>
        </div>
        <div class="divider-custom-line"></div>
      </div>

      <!-- Masthead Subheading -->
      <!-- <p class="masthead-subheading font-weight-light mb-0">Graphic Artist - Web Designer - Illustrator</p> -->
      <div class="upload-btn-wrapper mb-5">
        <?php echo
          "<form action='repeat.php' method='POST' enctype='multipart/form-data'>
            <h3 class='mb-3'>Testar imagem com outro método</h3>
              <div class='mb-3'>
                <select name='tecnica' class='btn'>";
                if($tecnica == "lbp"){
                  echo "
                  <option selected='selected' value='lbp'>LBP</option>
                  <option value='har'>Haralick</option>
                  <option value='hist'>Histograma de Cores</option>";
                }
                if($tecnica == "har"){
                  echo "
                  <option value='lbp'>LBP</option>
                  <option selected='selected' value='har'>Haralick</option>
                  <option value='hist'>Histograma de Cores</option>";
                }
                if($tecnica == "hist"){
                  echo "
                  <option value='lbp'>LBP</option>
                  <option value='har'>Haralick</option>
                  <option selected='selected' value='hist'>Histograma de Cores</option>";
                }
                echo "
                </select>
                <input type='hidden' name='image' value='".$image."'>
                <button class='btn text-uppercase' type='submit' name='submit'>Mudar método</button>
              </div>
            </form>";
          ?>
      </div>
      <div class="upload-btn-wrapper mb-5">
        <a class="btn" href="index.html">Retornar</a>
      </div>
    </div>
  </header>

  <!-- Footer -->
  <footer class="myfooter footer text-center align-items-center">
    <div class="row" id="grupo">
      <div class="col-lg-4 mb-5 mb-lg-0">
        <h2 class="text-uppercase mb-4"></h2>
      </div>
      <div class="col-lg-4 mb-5 mb-lg-0">
        <h2 class="text-uppercase mb-4">Grupo</h2>
        <p class="lead mb-0">Felipe Seppe de Faria
          <br>Luan Silveira Franco
          <br>Mateus Figueiredo Rego</p>
      </div>
      <div class="col-lg-4 mb-5 mb-lg-0">
        <h2 class="text-uppercase mb-4"></h2>
      </div>
    </div>
  </footer>

  <!-- Copyright Section -->
  <section class="copyright py-4 text-center text-white">
    <div class="container">
      <small>Copyright &copy; DeMorgam 2019</small>
    </div>
  </section>

  <!-- Scroll to Top Button (Only visible on small and extra-small screen sizes) -->
  <div class="scroll-to-top d-lg-none position-fixed ">
    <a class="js-scroll-trigger d-block text-center text-white rounded" href="#page-top">
      <i class="fa fa-chevron-up"></i>
    </a>
  </div>


  <!-- Bootstrap core JavaScript -->
  <script src="theme/vendor/jquery/jquery.min.js"></script>
  <script src="theme/vendor/bootstrap/js/bootstrap.bundle.min.js"></script>

  <!-- Plugin JavaScript -->
  <script src="theme/vendor/jquery-easing/jquery.easing.min.js"></script>

  <!-- Contact Form JavaScript -->
  <script src="theme/js/jqBootstrapValidation.js"></script>
  <script src="theme/js/contact_me.js"></script>

  <!-- Custom scripts for this template -->
  <script src="theme/js/freelancer.min.js"></script>

</body>

</html>
